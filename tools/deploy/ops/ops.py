import os

import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger


# ##### common functions ########################################


def export_onnx(model, args, f, input_names=None, output_names=None):
    # fill input_names and output_names
    assert isinstance(model, SimpleTracer), type(model)
    if input_names is None:
        input_names = model.get_input_names()
    if output_names is None:
        output_names = model.get_output_names()

    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            f,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=True,
            input_names=input_names,
            output_names=output_names,
        )


def get_inputs(*args, root="."):
    # load tensors from pth
    inputs = {}
    for name in args:
        assert isinstance(name, str), type(name)
        tensor = torch.load(os.path.join(root, "{}.pt".format(name)), map_location="cuda")
        inputs[name] = tensor
    return inputs


class SimpleTracer(torch.nn.Module):

    def __init__(self):
        super(SimpleTracer, self).__init__()

    def forward(self, inputs):
        assert isinstance(inputs, dict)
        outputs = self.inference(inputs)
        assert isinstance(outputs, dict)
        return outputs

    def inference(self, inputs):
        raise NotImplementedError

    def get_input_names(self):
        raise NotImplementedError

    def get_output_names(self):
        raise NotImplementedError


# ##### example: GenerateProposals ########################################


class GenerateProposals(SimpleTracer):

    def __init__(self):
        super(GenerateProposals, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        scores = inputs["scores"]
        bbox_deltas = inputs["bbox_deltas"]
        im_info = inputs["im_info"]
        cell_anchors_tensor = inputs["cell_anchors_tensor"]
        rpn_rois, rpn_roi_probs = torch.ops._caffe2.GenerateProposals(
            scores,
            bbox_deltas,
            im_info,
            cell_anchors_tensor,
            spatial_scale=0.25,
            pre_nms_topN=1000,
            post_nms_topN=100,  # 2000,
            nms_thresh=0.7,
            min_size=0,
            # correct_transform_coords=True,  # deprecated argument
            angle_bound_on=True,  # Default
            angle_bound_lo=-180,
            angle_bound_hi=180,
            clip_angle_thresh=1.0,  # Default
            legacy_plus_one=False,
        )
        return {
            "rpn_rois": rpn_rois,
            "rpn_roi_probs": rpn_roi_probs,
        }

    def get_input_names(self):
        return ["scores", "bbox_deltas", "im_info", "cell_anchors_tensor"]

    def get_output_names(self):
        return ["rpn_rois", "rpn_roi_probs"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: GenerateProposals")

    m = GenerateProposals()

    data = get_inputs("scores", "bbox_deltas", "im_info", "cell_anchors_tensor", root="./GenerateProposals")
    export_onnx(m, data, "model.onnx")
    targets = m(data)

    TensorRTModel.build_engine("model.onnx", "model.trt", 4, device="CUDA")
    e = TensorRTModel("model.trt")
    outputs = e.inference(data)

    # compare torch output and tensorrt output
    assert len(targets) == len(outputs), "Number of outputs does not match!"
    targets = [(k, v) for k, v in targets.items()]
    for i, (name, tensor) in enumerate(targets):
        logger.info(name)
        diff = outputs[i] - tensor
        unique = torch.unique(diff)
        logger.info("diff\n{}".format(diff))
        logger.info("unique\n{}".format(unique))
