import argparse
import functools
import logging
import os
import time
import types
from datetime import datetime

import torch
import torch.onnx
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models
import torchvision.transforms as transforms
from detectron2.export.meta_modeling import MetaModel, trace_context
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger


# ##### general helper module ########################################

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt

        # initialize meter
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)

    @property
    def err(self):
        return 100 - float(self.avg)

    def __str__(self):
        fmt_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmt_str.format(**self.__dict__)


class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmt_str = self._get_batch_fmt_str(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmt_str.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(datetime.now(), "\t".join(entries), flush=True)

    @staticmethod
    def _get_batch_fmt_str(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def validate(val_loader, model, cuda=True, print_freq=20):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5])

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if cuda:
                data = [t.cuda() for t in data]
            images, target = data
            # compute output
            # notice that this is slightly different from original torch model
            output = model(data)
            # measure and record accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
    print("Time {:.4f}".format(batch_time.avg), flush=True)
    print("Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5), flush=True)
    print("Err@1 {top1.err:.3f} Err@5 {top5.err:.3f}".format(top1=top1, top5=top5), flush=True)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def fill_trt_inputs(tensor, batch_size):
    zeros = torch.zeros(1, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
    padding = [zeros for _ in range(batch_size - tensor.size(0))]
    return torch.cat([tensor, *padding], dim=0).contiguous()


# ##### core element ########################################


class TorchModel(MetaModel):

    def __init__(self, torch_model):
        cfg = types.SimpleNamespace()
        super(TorchModel, self).__init__(cfg, torch_model)

    def convert_inputs(self, data):
        images, target = data
        return images.cuda()

    def convert_outputs(self, data, images, results):
        # no postprocessing step is needed for pytorch model
        return results

    def inference(self, inputs):
        # the naming is slightly different
        return self._wrapped_model(inputs)

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["logits"]


class TensorRTEngine(TensorRTModel, TorchModel):

    def __init__(self, engine_path, batch_size):
        super(TensorRTEngine, self).__init__(engine_path)
        TorchModel.__init__(self, self._engine)

        self.batch_size = batch_size
        self.effective_batch_size = 0

    def convert_inputs(self, data):
        images, target = data
        # TensorRT only supports fixed size inputs
        self.effective_batch_size = images.size(0)
        if self.effective_batch_size < self.batch_size:
            images = fill_trt_inputs(images, self.batch_size)
        return {"images": images.cuda()}

    def convert_outputs(self, data, inputs, results):
        assert len(results) == 1
        output = results[0]
        if self.effective_batch_size < self.batch_size:
            output = output[:self.effective_batch_size]
        return output


def get_data_loader(val_dir, batch_size, workers=2):
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return val_loader


def main():
    parser = argparse.ArgumentParser(description="ImageNet inference example")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 2)")
    parser.add_argument("-b", "--batch-size", default=1024, type=int,
                        metavar="N",
                        help="mini-batch size (default: 32), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--output", default="./output", help="output directory for the converted model")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calibration-batch", type=int, default=1024, help="max calibration batch number")
    parser.add_argument(
        "--format",
        choices=["torch", "onnx", "tensorrt"],
        help="output format",
        default="torch",
    )
    args = parser.parse_args()
    if args.debug:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    logger = setup_logger(verbosity=verbosity)
    logger.info("Command line arguments: " + str(args))

    if args.output:
        os.makedirs(args.output, exist_ok=True)
    suffix = "_fp16" if args.fp16 else ""
    suffix += "_int8" if args.int8 else ""
    onnx_f = os.path.join(args.output, "model.onnx")
    engine_f = os.path.join(args.output, "model{}.trt".format(suffix))
    cache_f = os.path.join(args.output, "cache.txt")

    # get data loader
    data_loader = get_data_loader(args.data, args.batch_size, args.workers)

    if args.format == "torch" or args.format == "onnx":
        torch_model = torchvision.models.resnet50(pretrained=True)
        torch_model.cuda()
        model = TorchModel(torch_model)
        if args.format == "onnx":
            data = next(iter(data_loader))
            inputs = model.convert_inputs(data)
            with trace_context(model), torch.no_grad():
                torch.onnx.export(model, (inputs,), onnx_f, verbose=True, input_names=model.get_input_names(),
                                  output_names=model.get_output_names())
                return
    else:
        if not os.path.exists(engine_f):
            if args.int8:
                ns = types.SimpleNamespace()
                ns.batch_size = args.batch_size
                ns.effective_batch_size = 0
                preprocess = functools.partial(TensorRTEngine.convert_inputs, ns)
                int8_calibrator = TensorRTModel.get_int8_calibrator(
                    args.calibration_batch, data_loader, preprocess, cache_f)
            else:
                int8_calibrator = None
            TensorRTModel.build_engine(onnx_f, engine_f, args.batch_size, device="CUDA",
                                       fp16_mode=args.fp16, int8_mode=args.int8, int8_calibrator=int8_calibrator)
        model = TensorRTEngine(engine_f, args.batch_size)
        model.cuda()

    # validation
    validate(data_loader, model)

    if args.format == "tensorrt":
        model.report_engine_time("engine_time.txt", 0.01)


if __name__ == "__main__":
    main()
