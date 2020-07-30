import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class CollectRpnProposals(SimpleTracer):

    def __init__(self):
        super(CollectRpnProposals, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        input_list = [v for k, v in inputs.items()]
        rpn_rois = torch.ops._caffe2.CollectRpnProposals(
            input_list,
            # NOTE: in current implementation, rpn_max_level and rpn_min_level
            # are not needed, only the subtraction of two matters and it
            # can be infer from the number of inputs. Keep them now for
            # consistency.
            rpn_max_level=6,
            rpn_min_level=2,
            rpn_post_nms_topN=200 * 1,  # 2000 * 1,
        )
        return {
            "rpn_rois": rpn_rois
        }

    def get_input_names(self):
        return ["rois_fpn2", "rois_fpn3", "rois_fpn4", "rois_fpn5", "rois_fpn6",
                "probs_fpn2", "probs_fpn3", "probs_fpn4", "probs_fpn5", "probs_fpn6"]

    def get_output_names(self):
        return ["rpn_rois"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: CollectRpnProposals")

    m = CollectRpnProposals()

    data = get_inputs("input_list",
                      root="/autox-sz/users/dongqixu/share/trt_plugins/CollectRpnProposals")

    # CollectRpnProposals does not support cuda backend
    data = {n: data["input_list"][i].cpu() for i, n in enumerate(m.get_input_names())}
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
        assert torch.abs(unique).max().item() < 1e-3
