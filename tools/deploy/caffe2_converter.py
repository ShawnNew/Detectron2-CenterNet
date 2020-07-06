#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import logging
import os
import re
import onnx
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import Caffe2Tracer, add_export_config
from detectron2.export.tensorrt import META_ARCH_TENSORRT_EXPORT_TYPE_MAP, TensorRTModel
from detectron2.modeling import build_model
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.logger import setup_logger


def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if cfg.MODEL.DEVICE != "cpu":
        assert TORCH_VERSION >= (1, 5), "PyTorch>=1.5 required for GPU conversion!"
    return cfg


def override(file):
    name = ""
    while not (len(name.strip()) == 1 and re.match("[YyNnQq]", name.strip())):
        print("File {} already exists. Override? [YyNnQq] ".format(file), end="")
        name = input()
    if name.lower() == "y":
        return True
    elif name.lower() == "q":
        exit(1)
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model using caffe2 tracing.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript", "tensorrt"],
        help="output format",
        default="caffe2",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calibration-batch", type=int, default=512, help="max calibration batch number")
    parser.add_argument("--calibration-cache", help="output calibration cache path")
    parser.add_argument("--quantization", help="input quantization layers definition")
    parser.add_argument("--exclude", help="input exclude quantization layers definition")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument("--onnx", help="output onnx model path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.debug:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    logger = setup_logger(verbosity=verbosity)
    logger.info("Command line arguments: " + str(args))
    os.makedirs(args.output, exist_ok=True)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    data_iter = iter(data_loader)
    first_batch = next(data_iter)

    # convert and save caffe2 model
    tracer = Caffe2Tracer(cfg, torch_model, first_batch)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=first_batch)
    elif args.format == "onnx":
        traceable_model, onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, args.onnx if args.onnx else os.path.join(args.output, "model.onnx"))
        del onnx_model
    elif args.format == "torchscript":
        script_model = tracer.export_torchscript()
        script_model.save(os.path.join(args.output, "model.ts"))

        # Recursively print IR of all modules
        with open(os.path.join(args.output, "model_ts_IR.txt"), "w") as f:
            try:
                f.write(script_model._actual_script_module._c.dump_to_str(True, False, False))
            except AttributeError:
                pass
        # Print IR of the entire graph (all submodules inlined)
        with open(os.path.join(args.output, "model_ts_IR_inlined.txt"), "w") as f:
            f.write(str(script_model.inlined_graph))
        # Print the model structure in pytorch style
        with open(os.path.join(args.output, "model.txt"), "w") as f:
            f.write(str(script_model))
    elif args.format == "tensorrt":
        suffix = "_fp16" if args.fp16 else ""
        suffix += "_int8" if args.int8 else ""
        onnx_f = args.onnx if args.onnx else os.path.join(args.output, "model.onnx")
        engine_f = os.path.join(args.output, "model{}.trt".format(suffix))
        cache_f = args.calibration_cache if args.calibration_cache else os.path.join(args.output, "cache.txt")
        assert os.path.isfile(onnx_f), "path {} is not a file".format(onnx_f)
        if not os.path.isfile(engine_f) or (not args.cache and override(engine_f)):
            if args.int8:
                # get preprocess function from model
                model = tracer.get_onnx_traceable()
                max_calibration_batch = args.calibration_batch
                if os.path.exists(cache_f) and (not args.cache and override(cache_f)):
                    os.remove(cache_f)
                int8_calibrator = TensorRTModel.get_int8_calibrator(
                    max_calibration_batch, data_loader, model.convert_inputs, cache_f)
                if args.quantization:
                    quantization_layers = []
                    with open(args.quantization) as f:
                        for line in f:
                            quantization_layers.append(line.strip())
                    logger.info("quantization_layers: {}".format(quantization_layers))
                else:
                    quantization_layers = None
                if args.exclude:
                    assert not args.quantization, "exclude and quantization cannot be set simultaneously."
                    exclude_layers = []
                    with open(args.exclude) as f:
                        for line in f:
                            exclude_layers.append(line.strip())
                    logger.info("exclude_layers: {}".format(exclude_layers))
                else:
                    exclude_layers = None
            else:
                int8_calibrator = None
                quantization_layers = None
                exclude_layers = None
            TensorRTModel.build_engine(onnx_f, engine_f, cfg.TEST.BATCH_SIZE, device=cfg.MODEL.DEVICE.upper(),
                                       fp16_mode=args.fp16, int8_mode=args.int8, int8_calibrator=int8_calibrator,
                                       quantization_layers=quantization_layers, exclude_layers=exclude_layers)
            # release data iter
            if int8_calibrator is not None:
                del int8_calibrator

    # GC
    del first_batch
    del data_iter
    del data_loader
    del tracer
    del torch_model
    torch.cuda.empty_cache()

    # run evaluation with the converted model
    if args.run_eval:
        if args.format == "onnx":
            model = traceable_model
        elif args.format == "tensorrt":
            MetaArch = META_ARCH_TENSORRT_EXPORT_TYPE_MAP[cfg.MODEL.META_ARCHITECTURE]
            model = MetaArch(cfg, engine_f)
            model.to(torch.device(cfg.MODEL.DEVICE))
        else:
            model = caffe2_model
            assert args.format == "caffe2", "Python inference in other format is not yet supported."
        logger.info("Inference traceable model\n{}".format(str(model)))
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, cfg, True, args.output)
        metrics = inference_on_dataset(model, data_loader, evaluator)
        print_csv_format(metrics)
        if args.format == "tensorrt":
            if args.fp16 or args.int8:
                threshold = 0.1
            else:
                threshold = 0.5
            engine_time = model.report_engine_time("engine_time.txt", threshold)
            # write performance to file
            perf_f = os.path.join(args.output, "perf.txt")
            with open(perf_f, "w") as f:
                f.write("engine {}\n".format(os.path.abspath(engine_f)))
                f.write("time {}\n".format(engine_time))
                for k, v in metrics.items():
                    f.write("{} {}\n".format(k, v["AP"]))
