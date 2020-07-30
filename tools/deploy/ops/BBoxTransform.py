import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class BBoxTransform(SimpleTracer):

    def __init__(self):
        super(BBoxTransform, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        rois = inputs["rois"]
        box_regression = inputs["box_regression"]
        im_info = inputs["im_info"]
        roi_pred_bbox, roi_batch_splits = torch.ops._caffe2.BBoxTransform(
            rois,
            box_regression,
            im_info,
            weights=(10.0, 10.0, 5.0, 5.0),
            apply_scale=True,
            rotated=False,
            angle_bound_on=True,
            angle_bound_lo=-180,
            angle_bound_hi=180,
            clip_angle_thresh=1.0,
            legacy_plus_one=False,
        )
        return {
            "roi_pred_bbox": roi_pred_bbox,
            "roi_batch_splits": roi_batch_splits,
        }

    def get_input_names(self):
        return ["rois", "box_regression", "im_info"]

    def get_output_names(self):
        return ["roi_pred_bbox", "roi_batch_splits"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: BBoxTransform")

    m = BBoxTransform()

    data = get_inputs("rois", "box_regression", "im_info",
                      root="/autox-sz/users/dongqixu/share/trt_plugins/BBoxTransform", map_location="cpu")
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
