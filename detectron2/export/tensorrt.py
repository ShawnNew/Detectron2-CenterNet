import functools
import logging
import re
import time
import types

import onnx
import tensorrt as trt
import torch
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling import meta_arch
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures import Boxes, Instances, RotatedBoxes
from termcolor import colored

from .meta_modeling import MetaModel, RetinaNetModel, ProposalModel, GeneralizedRCNNModel, CenterNetModel
from .onnx_tensorrt import backend, to_cuda
from .onnx_tensorrt.calibrator import PythonEntropyCalibrator
from .onnx_tensorrt.tensorrt_engine import Engine

logger = logging.getLogger(__name__)


def to_numpy(tensor):
    """
    Convert tensor to numpy array.

    Return:
        numpy.ndarray
    """
    if tensor.dtype == torch.int64:
        tensor = tensor.to(dtype=torch.int)
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.cpu().numpy()


class TensorRTModel:
    """
    Base class for TensorRT inference implementation of a meta architecture.
    """

    def __init__(self, engine_path):
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
        if not trt.init_libnvinfer_plugins(trt_logger, ""):
            raise RuntimeError("Failed to initialize TensorRT's plugin library.")
        self._runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            engine = self._runtime.deserialize_cuda_engine(f.read())
        self._engine = Engine(engine)
        logger.info("Load TensorRT engine {}".format(engine_path))

    def inference(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return self._engine.run([to_cuda(inputs)])
        elif isinstance(inputs, list):
            return self._engine.run([to_cuda(v) for v in inputs])
        elif isinstance(inputs, dict):
            m_inputs = {}
            for i, binding in enumerate(self._engine.inputs):
                name = re.sub("^__", "", binding.name)
                assert name in inputs, name
                m_inputs[name] = inputs[name]
            m_results = self._engine.run([to_cuda(v) for k, v in m_inputs.items()])
            return {binding.name: m_results[i] for i, binding in enumerate(self._engine.outputs)}
        else:
            raise NotImplementedError

    def report_engine_time(self, filename: str, threshold: float):
        return self._engine.report_engine_time(filename, threshold)

    @classmethod
    def build_engine(cls, onnx_f, engine_f, max_batch_size, max_workspace_size=None, device=None,
                     fp16_mode=False, int8_mode=False, int8_calibrator=None, quantization_layers=None,
                     exclude_layers=None):
        if fp16_mode:
            logger.info(colored("set fp16 mode enabled", "blue"))
        if int8_mode:
            assert int8_calibrator is not None, "Calibrator is not set with int8 mode used."
            logger.info(colored("set int8 mode enabled", "blue"))
        assert device is not None, device

        onnx_model = onnx.load(onnx_f)
        onnx.checker.check_model(onnx_model)

        if max_workspace_size is None:
            max_workspace_size = 6 << 30
        start_time = time.perf_counter()
        engine = backend.prepare(onnx_model, device, max_batch_size=max_batch_size,
                                 max_workspace_size=max_workspace_size, serialize_engine=True,
                                 fp16_mode=fp16_mode, int8_mode=int8_mode, int8_calibrator=int8_calibrator,
                                 quantization_layers=quantization_layers, exclude_layers=exclude_layers)
        total_time = time.perf_counter() - start_time
        logger.info("Engine build time: {:.2f} s".format(total_time))
        with open(engine_f, "wb") as f:
            engine = engine.engine.engine
            f.write(engine.serialize())
        logger.info("TensorRT engine is saved to {}".format(engine_f))

    @classmethod
    def get_int8_calibrator(cls, max_calibration_batch, data_loader, preprocess_f, cache_file):
        return PythonEntropyCalibrator(max_calibration_batch, data_loader, preprocess_f, cache_file)


class TensorRTRetinaNet(TensorRTModel, RetinaNetModel):
    def __init__(self, cfg, engine_path):
        super(TensorRTRetinaNet, self).__init__(engine_path)
        RetinaNetModel.__init__(self, cfg, self._engine)

        # preprocess parameters
        ns = types.SimpleNamespace()
        ns.training = False
        ns.input = self._cfg.INPUT
        ns.dynamic = self._cfg.INPUT.DYNAMIC
        ns.device = torch.device(self._cfg.MODEL.DEVICE)
        ns.pixel_mean = torch.tensor(self._cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(ns.device)
        ns.pixel_std = torch.tensor(self._cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(ns.device)

        ns.backbone = types.SimpleNamespace()
        ns.backbone.size_divisibility = 32

        # inference parameters
        ns.num_classes = self._cfg.MODEL.RETINANET.NUM_CLASSES
        ns.topk_candidates = self._cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        ns.score_threshold = self._cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        ns.nms_threshold = self._cfg.MODEL.RETINANET.NMS_THRESH_TEST
        ns.max_detections_per_image = self._cfg.TEST.DETECTIONS_PER_IMAGE

        # anchor generator
        feature_shapes = [ShapeSpec(stride=s) for s in (8, 16, 32, 64, 128)]
        self._anchor_generator = build_anchor_generator(self._cfg, feature_shapes)

        ns.preprocess_image = functools.partial(meta_arch.RetinaNet.preprocess_image, ns)
        ns.inference = functools.partial(meta_arch.RetinaNet.inference, ns)
        ns.inference_single_image = functools.partial(meta_arch.RetinaNet.inference_single_image, ns)
        ns.box2box_transform = Box2BoxTransform(weights=self._cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

        self._ns = ns

    def convert_inputs(self, batched_inputs):
        images = self._ns.preprocess_image(batched_inputs)
        return {
            "images": images.tensor,
            "image_sizes": torch.tensor(images.image_sizes),
        }

    def convert_outputs(self, batched_inputs, inputs, results):
        output_names = self.get_output_names()
        assert len(results) == len(output_names)

        m_results = {}
        for k, v in results.items():
            assert k in output_names, k
            m_results[k] = v.to(self._ns.device)

        image_sizes = inputs["image_sizes"]

        num_features = len([x for x in m_results.keys() if x.startswith("box_cls_")])
        pred_logits = [m_results["box_cls_{}".format(i)] for i in range(num_features)]
        pred_anchor_deltas = [m_results["box_delta_{}".format(i)] for i in range(num_features)]

        # generate anchors from anchor_generator
        anchors = self._anchor_generator(pred_logits)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self._ns.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        results = self._ns.inference(anchors, pred_logits, pred_anchor_deltas, image_sizes)
        return meta_arch.GeneralizedRCNN._postprocess(results, batched_inputs, image_sizes)


class TensorRTGeneralizedRCNN(TensorRTModel, GeneralizedRCNNModel):

    def __init__(self, cfg, engine_path):
        super(TensorRTGeneralizedRCNN, self).__init__(engine_path)
        MetaModel.__init__(self, cfg, self._engine)

        # preprocess parameters
        ns = types.SimpleNamespace()
        ns.training = False
        ns.input = self._cfg.INPUT
        ns.dynamic = self._cfg.INPUT.DYNAMIC
        ns.device = torch.device(self._cfg.MODEL.DEVICE)
        ns.pixel_mean = torch.tensor(self._cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(ns.device)
        ns.pixel_std = torch.tensor(self._cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(ns.device)
        # size_divisibility
        ns.backbone = types.SimpleNamespace()
        ns.backbone.size_divisibility = 32

        ns.preprocess_image = functools.partial(meta_arch.GeneralizedRCNN.preprocess_image, ns)
        self._ns = ns

        # reuse convert_inputs defined in TensorRTProposal
        self._convert_inputs = functools.partial(TensorRTProposal.convert_inputs, self)

    def convert_inputs(self, batched_inputs):
        return self._convert_inputs(batched_inputs)

    def convert_outputs(self, batched_inputs, inputs, results):
        output_names = self.get_output_names()
        assert len(results) == len(output_names)

        m_results = {}
        for k, v in results.items():
            assert k in output_names, k
            m_results[k] = v.to(self._ns.device)

        # TensorRT output number is not dynamic
        image_sizes = inputs["image_sizes"]
        m_instances = [Instances(image_size) for image_size in image_sizes]

        # pred_boxes format: (batch_index, x0, y0, x1, y1)
        pred_boxes = m_results["pred_boxes"][:, 1:]
        scores = m_results["scores"]
        pred_classes = m_results["pred_classes"].to(torch.int64)
        batch_splits = m_results["batch_splits"].to(torch.int64).cpu()
        pred_masks = m_results.get("pred_masks", None)
        if pred_boxes.shape[1] == 5:
            pred_boxes = RotatedBoxes(pred_boxes)
        else:
            pred_boxes = Boxes(pred_boxes)

        offset = 0
        for i in range(len(batched_inputs)):
            next_offset = offset + batch_splits[i]
            m_instances[i].pred_boxes = pred_boxes[offset:next_offset]
            m_instances[i].scores = scores[offset:next_offset]
            m_instances[i].pred_classes = pred_classes[offset:next_offset]
            if "pred_masks" in m_results:
                num_masks = batch_splits[i]
                indices = torch.arange(num_masks, device=pred_classes.device)
                m_instances[i].pred_masks = \
                    pred_masks[offset:next_offset][indices, m_instances[i].pred_classes][:, None]
            offset += int(len(pred_boxes) / len(batched_inputs))

        return meta_arch.GeneralizedRCNN._postprocess(m_instances, batched_inputs, image_sizes)


class TensorRTProposal(TensorRTModel, ProposalModel):

    def __init__(self, cfg, engine_path):
        super(TensorRTProposal, self).__init__(engine_path)
        MetaModel.__init__(self, cfg, self._engine)

        # preprocess parameters
        ns = types.SimpleNamespace()
        ns.training = False
        ns.input = self._cfg.INPUT
        ns.dynamic = self._cfg.INPUT.DYNAMIC
        ns.device = torch.device(self._cfg.MODEL.DEVICE)
        ns.pixel_mean = torch.tensor(self._cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(ns.device)
        ns.pixel_std = torch.tensor(self._cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(ns.device)
        # size_divisibility
        ns.backbone = types.SimpleNamespace()
        ns.backbone.size_divisibility = 32

        ns.preprocess_image = functools.partial(meta_arch.GeneralizedRCNN.preprocess_image, ns)
        self._ns = ns

    def convert_inputs(self, batched_inputs):
        images = self._ns.preprocess_image(batched_inputs)

        # compute scales and im_info
        assert not self._ns.training
        min_size = self._ns.input.MIN_SIZE_TEST
        max_size = self._ns.input.MAX_SIZE_TEST
        min_size = min_size[0] if isinstance(min_size, tuple) else min_size
        scales = []
        for i in range(len(batched_inputs)):
            s = max(batched_inputs[i]["height"] * 1.0 / min_size,
                    batched_inputs[i]["width"] * 1.0 / max_size)
            scales.append((s,))
        im_info = []
        for image_size, scale in zip(images.image_sizes, scales):
            im_info.append([*image_size, *scale])
        return {
            "images": images.tensor,
            "image_sizes": torch.tensor(images.image_sizes),
            "im_info": torch.tensor(im_info, device=images.tensor.device),
        }

    def convert_outputs(self, batched_inputs, inputs, results):
        output_names = self.get_output_names()
        assert len(results) == len(output_names)

        m_results = {}
        for k, v in results.items():
            assert k in output_names, k
            m_results[k] = v.to(self._ns.device)
        return super(TensorRTProposal, self).convert_outputs(batched_inputs, inputs, m_results)

class TensorRTCenterNet(TensorRTModel, CenterNetModel):

    def __init__(self, cfg, engine_path):
        super(TensorRTCenterNet, self).__init__(engine_path)
        MetaModel.__init__(self, cfg, self._engine)

        # preprocess parameters
        ns = types.SimpleNamespace()
        ns.training = False
        ns.input = cfg.INPUT
        # ns.dynamic = cfg.INPUT.DYNAMIC
        ns.device = torch.device(cfg.MODEL.DEVICE)
        ns.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(ns.device)
        ns.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(ns.device)

        ns.preprocess_image = functools.partial(meta_arch.CenterNet.preprocess_image, ns)
        ns.inference = functools.partial(meta_arch.CenterNet.inference, ns)
        ns.inference_single_image = functools.partial(meta_arch.CenterNet.inference_single_image, ns)
        ns.topk_candidates = cfg.MODEL.CENTERNET.TOPK_CANDIDATES_TEST
        ns.score_threshold = cfg.MODEL.CENTERNET.SCORE_THRESH_TEST
        ns.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE


        # size_divisibility
        ns.backbone = types.SimpleNamespace()
        ns.backbone.size_divisibility = 16
        ns.size_divisibility = 16
        ns.backbone.down_ratio = 4
        self._ns = ns

    def convert_inputs(self, batched_inputs):
        images, _ = self._ns.preprocess_image(batched_inputs)

        # compute scales and im_info
        assert not self._ns.training
        return {
            "images": images.tensor,
            "im_info": torch.tensor(images.image_sizes),
        }

    def convert_outputs(self, batched_inputs, inputs, results):
        output_names = self.get_output_names()
        assert len(results) == len(output_names)
        results = self._ns.inference(results, inputs['im_info'])

        from detectron2.modeling.postprocessing import detector_postprocess
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, inputs['im_info']
        ):
            original_height = input_per_image.get("height", image_size[0])
            original_width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, original_height, original_width)
            processed_results.append({"instances": r})
        return processed_results



META_ARCH_TENSORRT_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": TensorRTGeneralizedRCNN,
    "RetinaNet": TensorRTRetinaNet,
    "ProposalNetwork": TensorRTProposal,
    "CenterNet": TensorRTCenterNet
}
