import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class DistributeFpnProposals(SimpleTracer):

    def __init__(self):
        super(DistributeFpnProposals, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        pooler_fmt_boxes = inputs["pooler_fmt_boxes"]
        fpn_outputs = torch.ops._caffe2.DistributeFpnProposals(
            pooler_fmt_boxes,
            roi_canonical_scale=224,
            roi_canonical_level=4,
            roi_max_level=5,
            roi_min_level=2,
            legacy_plus_one=False,
        )
        assert len(fpn_outputs) == len(self.get_output_names())
        return {k: fpn_outputs[i] for i, k in enumerate(self.get_output_names())}

    def get_input_names(self):
        return ["pooler_fmt_boxes"]

    def get_output_names(self):
        return ["rois_fpn2", "rois_fpn3", "rois_fpn4", "rois_fpn5", "rois_idx_restore"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: DistributeFpnProposals")

    m = DistributeFpnProposals()

    data = get_inputs("pooler_fmt_boxes",
                      root="/autox-sz/users/dongqixu/share/trt_plugins/DistributeFpnProposals", map_location="cpu")
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
        # dynamic output size
        length = tensor.size(0)
        diff = outputs[name][:length] - tensor
        remain = outputs[name][length:]
        unique = torch.unique(diff)
        logger.info("unique\n{}".format(unique))
        logger.info("max\n{}".format(torch.abs(unique).max()))
        logger.info("remain\n{}".format(torch.unique(remain)))

        assert torch.abs(unique).max() < 1e-3
        assert len(torch.unique(remain)) == 0 or \
            len(torch.unique(remain)) == 1 and torch.unique(remain).item() == -1
