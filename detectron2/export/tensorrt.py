import functools
import logging
import os
import sys
import types

import tensorrt as trt
import torch

from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling import meta_arch
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from .meta_modeling import RetinaNetModel

logger = logging.getLogger(__name__)
print("load onnx_tensorrt from {}".format(os.environ["onnx_tensorrt_root"]))

sys.path.append(os.environ["onnx_tensorrt_root"])
from onnx_tensorrt.tensorrt_engine import Engine


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
            return self._engine.run([to_numpy(inputs)])
        elif isinstance(inputs, list):
            return self._engine.run([to_numpy(v) for v in inputs])
        elif isinstance(inputs, dict):
            return self._engine.run([to_numpy(v) for k, v in inputs.items()])
        else:
            raise NotImplementedError


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

        # convert TensorRT output to tensor
        m_results = {}
        for i, binding in enumerate(self._engine.outputs):
            assert binding.name in output_names, binding.name
            if binding.name == "image_sizes":
                continue
            m_results[binding.name] = torch.from_numpy(results[i]).to(self._ns.device)

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


META_ARCH_TENSORRT_EXPORT_TYPE_MAP = {
    "RetinaNet": TensorRTRetinaNet,
}
