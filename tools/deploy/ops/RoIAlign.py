import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class RoIAlign(SimpleTracer):

    def __init__(self):
        super(RoIAlign, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        x_level = inputs["x_level"]
        roi_fpn = inputs["roi_fpn"]
        roi_feat_fpn = torch.ops._caffe2.RoIAlign(
            x_level,
            roi_fpn,
            order="NCHW",
            spatial_scale=0.25,
            pooled_h=7,
            pooled_w=7,
            sampling_ratio=0,
            aligned=True,
        )
        return {
            "roi_feat_fpn": roi_feat_fpn,
        }

    def get_input_names(self):
        return ["x_level", "roi_fpn"]

    def get_output_names(self):
        return ["roi_feat_fpn"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: RoIAlign")

    m = RoIAlign()

    data = get_inputs("x_level", "roi_fpn", root="/autox-sz/users/dongqixu/share/trt_plugins/RoIAlign")
    export_onnx(m, data, "model.onnx")
    targets = m(data)

    TensorRTModel.build_engine("model.onnx", "model.trt", 4, device="CUDA")
    e = TensorRTModel("model.trt")
    outputs = e.inference(data)

    # compare torch output and tensorrt output
    assert len(targets) == len(outputs), "Number of outputs does not match!"
    targets = [(k, v.cuda()) for k, v in targets.items()]
    for i, (name, tensor) in enumerate(targets):
        logger.info(name)
        diff = outputs[name] - tensor
        unique = torch.unique(diff)
        logger.info("unique\n{}".format(unique))
        logger.info("max\n{}".format(torch.abs(unique).max()))
        assert torch.abs(unique).max() < 1e-3
