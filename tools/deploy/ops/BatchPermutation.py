import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class BatchPermutation(SimpleTracer):

    def __init__(self):
        super(BatchPermutation, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        roi_feat_shuffled = inputs["roi_feat_shuffled"]
        rois_idx_restore_int32 = inputs["rois_idx_restore_int32"]
        roi_feat = torch.ops._caffe2.BatchPermutation(roi_feat_shuffled, rois_idx_restore_int32)
        return {
            "roi_feat": roi_feat,
        }

    def get_input_names(self):
        return ["roi_feat_shuffled", "rois_idx_restore_int32"]

    def get_output_names(self):
        return ["roi_feat"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: BatchPermutation")

    m = BatchPermutation()

    data = get_inputs("roi_feat_shuffled", "rois_idx_restore_int32",
                      root="/autox-sz/users/dongqixu/share/trt_plugins/BatchPermutation")
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
