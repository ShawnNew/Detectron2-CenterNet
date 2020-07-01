import contextlib
import logging
import time

import onnx
import torch

from detectron2.modeling import meta_arch
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.structures import Boxes, Instances, RotatedBoxes

from .patcher import ROIHeadsPatcher, patch_generalized_rcnn

logger = logging.getLogger(__name__)


class MetaModel(torch.nn.Module):
    """
    Base class for trace compatible implementation of a meta architecture.
    The forward can be set as trace mode to support onnx export.
    """

    def __init__(self, cfg, torch_model, trace_mode=False):
        """
        Args:
            cfg (CfgNode):
            torch_model (nn.Module): the detectron2 model (meta_arch) to be
                converted.
            trace_mode (bool):
        """
        super(MetaModel, self).__init__()
        self._cfg = cfg
        self._wrapped_model = torch_model
        self._trace_mode = trace_mode
        self.eval()

    def convert_inputs(self, batched_inputs):
        """
        Convert inputs from data loader to preprocessed images to model inputs.

        Args:
            batched_inputs (list[dict]): inputs to a detectron2 model
                in its standard format. Each dict has "image" (CHW tensor), and optionally
                "height" and "width".
        """
        raise NotImplementedError

    def convert_outputs(self, batched_inputs, inputs, results):
        """
        Convert inference results to post-processed outputs.

        Args:
            batched_inputs (list[dict]): the original input format of the meta arch
            inputs (dict[str, Tensor]): the model inputs.
            results (dict[str, Tensor]): the model results.
        """
        raise NotImplementedError

    def inference(self, inputs):
        """
        The core logic of model inference.

        Args:
            inputs (dict[str, Tensor]): the model inputs.
        """
        raise NotImplementedError

    def forward(self, inputs):
        """
        Run the forward in trace compatible mode.

        Args:
            inputs (list[dict] or dict[str, Tensor]):
                the original input format of the meta arch or the model inputs.
        """
        if self._trace_mode:
            return self.inference(inputs)
        with Timer() as preprocess:
            m_inputs = self.convert_inputs(inputs)
        with Timer() as inference:
            m_results = self.inference(m_inputs)
        with Timer() as postprocess:
            m_outputs = self.convert_outputs(inputs, m_inputs, m_results)
        logger.debug("preprocess: {:6.2f} ms,  inference: {:6.2f} ms, postprocess: {:6.2f} ms".format(
            preprocess.time, inference.time, postprocess.time))
        return m_outputs

    def get_input_names(self):
        raise NotImplementedError

    def get_output_names(self):
        raise NotImplementedError


class RetinaNetModel(MetaModel):

    def __init__(self, cfg, torch_model, trace_mode=False):
        super(RetinaNetModel, self).__init__(cfg, torch_model, trace_mode)

    def convert_inputs(self, batched_inputs):
        assert isinstance(self._wrapped_model, meta_arch.RetinaNet)
        images = self._wrapped_model.preprocess_image(batched_inputs)
        return {
            "images": images.tensor,
            "image_sizes": torch.tensor(images.image_sizes),
        }

    def convert_outputs(self, batched_inputs, inputs, results):
        assert isinstance(self._wrapped_model, meta_arch.RetinaNet)
        image_sizes = inputs["image_sizes"]

        num_features = len([x for x in results.keys() if x.startswith("box_cls_")])
        pred_logits = [results["box_cls_{}".format(i)] for i in range(num_features)]
        pred_anchor_deltas = [results["box_delta_{}".format(i)] for i in range(num_features)]

        # generate anchors from wrapped_model anchor_generator
        anchors = self._wrapped_model.anchor_generator(pred_logits)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self._wrapped_model.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]
        results = self._wrapped_model.inference(anchors, pred_logits, pred_anchor_deltas, image_sizes)
        return meta_arch.GeneralizedRCNN._postprocess(results, batched_inputs, image_sizes)

    def inference(self, inputs):
        assert isinstance(self._wrapped_model, meta_arch.RetinaNet)
        images = inputs["images"]
        features = self._wrapped_model.backbone(images)
        features = [features[f] for f in self._wrapped_model.in_features]
        pred_logits, pred_anchor_deltas = self._wrapped_model.head(features)

        results = {"image_sizes": inputs["image_sizes"]}
        for i, (box_cls_i, box_delta_i) in enumerate(zip(pred_logits, pred_anchor_deltas)):
            results["box_cls_{}".format(i)] = box_cls_i
            results["box_delta_{}".format(i)] = box_delta_i
        return results

    def get_input_names(self):
        return ["images", "image_sizes"]

    def get_output_names(self):
        num_features = 5
        output_names = []
        for i in range(num_features):
            output_names.append("box_cls_{}".format(i))
            output_names.append("box_delta_{}".format(i))
        return ["image_sizes"] + output_names


class GeneralizedRCNNModel(MetaModel):

    def __init__(self, cfg, torch_model, trace_mode=False):
        assert isinstance(torch_model, meta_arch.GeneralizedRCNN)
        torch_model = patch_generalized_rcnn(torch_model)
        torch_model.proposal_generator.tensor_mode = True
        super(GeneralizedRCNNModel, self).__init__(cfg, torch_model, trace_mode)

        self._roi_heads_patcher = ROIHeadsPatcher(cfg, self._wrapped_model.roi_heads, caffe2=False)

    def convert_inputs(self, batched_inputs):
        assert isinstance(self._wrapped_model, meta_arch.GeneralizedRCNN)
        images = self._wrapped_model.preprocess_image(batched_inputs)

        # compute scales and im_info
        assert not self._wrapped_model.training
        min_size = self._wrapped_model.input.MIN_SIZE_TEST
        max_size = self._wrapped_model.input.MAX_SIZE_TEST
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
        image_sizes = inputs["image_sizes"]
        m_results = [Instances(image_size) for image_size in image_sizes]

        pred_boxes = results["pred_boxes"]
        scores = results["scores"]
        pred_classes = results["pred_classes"].to(torch.int64)
        batch_splits = results["batch_splits"].to(torch.int64).cpu()
        pred_masks = results.get("pred_masks", None)
        if pred_boxes.shape[1] == 5:
            pred_boxes = RotatedBoxes(pred_boxes)
        else:
            pred_boxes = Boxes(pred_boxes)

        offset = 0
        for i in range(len(batched_inputs)):
            next_offset = offset + batch_splits[i]
            m_results[i].pred_boxes = pred_boxes[offset:next_offset]
            m_results[i].scores = scores[offset:next_offset]
            m_results[i].pred_classes = pred_classes[offset:next_offset]
            if "pred_masks" in results:
                num_masks = batch_splits[i]
                indices = torch.arange(num_masks, device=pred_classes.device)
                m_results[i].pred_masks = \
                    pred_masks[offset:next_offset][indices, m_results[i].pred_classes][:, None]
            offset = next_offset

        return meta_arch.GeneralizedRCNN._postprocess(m_results, batched_inputs, image_sizes)

    def inference(self, inputs):
        images = inputs["images"]
        features = self._wrapped_model.backbone(images)
        proposals, _ = self._wrapped_model.proposal_generator(inputs, features)
        with self._roi_heads_patcher.mock_roi_heads():
            detector_results, _ = self._wrapped_model.roi_heads(inputs, features, proposals)
        flatten = detector_results[0].flatten()

        results = {}
        for i, output_name in enumerate(detector_results[0].batch_extra_fields.keys()):
            results[output_name] = flatten[i]
        return results

    def get_input_names(self):
        return ["images", "image_sizes", "im_info"]

    def get_output_names(self):
        if hasattr(self._wrapped_model.roi_heads, "mask_pooler") or \
                hasattr(self._wrapped_model.roi_heads, "mask_head"):
            return ["pred_boxes", "scores", "pred_classes", "batch_splits", "pred_masks"]
        else:
            return ["pred_boxes", "scores", "pred_classes", "batch_splits"]


class Timer:
    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = (time.perf_counter() - self.time) * 1000


@contextlib.contextmanager
def trace_context(model):
    """
    A context where the model is temporarily changed to trace mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    trace_mode = model._trace_mode
    model._trace_mode = True
    yield
    model._trace_mode = trace_mode


def remove_copy_between_cpu_and_gpu(onnx_model):
    """
    In-place remove copy ops between cpu/gpu in onnx model.
    """

    def _rename_node_input_output(_node, _src, _dst):
        for _i, name in enumerate(_node.input):
            if name == _src:
                logger.info("rename {} input: {} -> {}".format(_node.name, _src, _dst))
                _node.input[_i] = _dst
        for _i, name in enumerate(_node.output):
            if name == _src:
                logger.info("rename {} output: {} -> {}".format(_node.name, _src, _dst))
                _node.output[_i] = _dst

    _COPY_OPS = ["CopyCPUToGPU", "CopyGPUToCPU"]

    onnx.checker.check_model(onnx_model)
    remove_list = []
    name_pairs = []
    for node in onnx_model.graph.node:
        if node.op_type in _COPY_OPS:
            assert len(node.input) == len(node.output) == 1
            name_pairs.append([node.input[0], node.output[0]])
            remove_list.append(node)

    if not remove_list:
        return onnx_model

    # maintain a list of set where identical input and output are put together
    # and an indexing mapping for each input and output
    identities = []
    node_index = {}
    for src, dst in name_pairs:
        src_i = node_index.get(src, -1)
        dst_i = node_index.get(dst, -1)
        if src_i < 0 and dst_i < 0:
            node_index[src] = len(identities)
            node_index[dst] = len(identities)
            identities.append({src, dst})
        elif src_i > 0 and dst_i < 0:
            node_index[dst] = node_index[src]
            identities[src_i].add(dst)
        elif src_i < 0 and dst_i > 0:
            node_index[src] = node_index[dst]
            identities[dst_i].add(src)

    # keep onnx model bindings unchanged
    model_bindings = set.union(
        set([node.name for node in onnx_model.graph.input]),
        set([node.name for node in onnx_model.graph.output]))

    for i, identity in enumerate(identities):
        binding = set.intersection(identity, model_bindings)
        if binding:
            assert len(binding) == 1, binding
            identity = binding
        identities[i] = sorted(list(identity))[-1]

    for node in remove_list:
        logger.info("remove {}: {} -> {}".format(node.name, node.input[0], node.output[0]))
        onnx_model.graph.node.remove(node)
    for node in onnx_model.graph.node:
        for src, idx in node_index.items():
            dst = identities[idx]
            if src != dst:
                _rename_node_input_output(node, src, dst)

    onnx.checker.check_model(onnx_model)
    return onnx_model


META_ARCH_ONNX_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": GeneralizedRCNNModel,
    "RetinaNet": RetinaNetModel,
}
