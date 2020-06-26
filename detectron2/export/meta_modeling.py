import contextlib

import torch

from detectron2.modeling import meta_arch
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K


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
        m_inputs = self.convert_inputs(inputs)
        m_results = self.inference(m_inputs)
        m_outputs = self.convert_outputs(inputs, m_inputs, m_results)
        return m_outputs


class RetinaNetModel(MetaModel):

    def __init__(self, cfg, torch_model, trace_mode=False):
        assert isinstance(torch_model, meta_arch.RetinaNet)
        super(RetinaNetModel, self).__init__(cfg, torch_model, trace_mode)

    def convert_inputs(self, batched_inputs):
        images = self._wrapped_model.preprocess_image(batched_inputs)
        return {
            "images": images.tensor,
            "image_sizes": torch.tensor(images.image_sizes),
        }

    def convert_outputs(self, batched_inputs, inputs, results):
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
        images = inputs["images"]
        features = self._wrapped_model.backbone(images)
        features = [features[f] for f in self._wrapped_model.in_features]
        pred_logits, pred_anchor_deltas = self._wrapped_model.head(features)

        results = {}
        for i, (box_cls_i, box_delta_i) in enumerate(zip(pred_logits, pred_anchor_deltas)):
            results["box_cls_{}".format(i)] = box_cls_i
            results["box_delta_{}".format(i)] = box_delta_i
        return results


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


META_ARCH_ONNX_EXPORT_TYPE_MAP = {
    "RetinaNet": RetinaNetModel,
}
