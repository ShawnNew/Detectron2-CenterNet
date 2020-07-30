import torch
import torch.nn
from detectron2.export.tensorrt import TensorRTModel
from detectron2.utils.logger import setup_logger

from ops import export_onnx, get_inputs, SimpleTracer


class BoxWithNMSLimit(SimpleTracer):

    def __init__(self):
        super(BoxWithNMSLimit, self).__init__()
        self.cuda()
        self.eval()

    def inference(self, inputs):
        class_prob = inputs["class_prob"]
        roi_pred_bbox = inputs["roi_pred_bbox"]
        roi_batch_splits = inputs["roi_batch_splits"]
        nms_outputs = torch.ops._caffe2.BoxWithNMSLimit(
            class_prob,
            roi_pred_bbox,
            roi_batch_splits,
            score_thresh=float(0.05),
            nms=float(0.5),
            detections_per_im=int(100),
            soft_nms_enabled=False,
            soft_nms_method="linear",
            soft_nms_sigma=0.5,
            soft_nms_min_score_thres=0.001,
            rotated=False,
            cls_agnostic_bbox_reg=False,
            input_boxes_include_bg_cls=False,
            output_classes_include_bg_cls=False,
            legacy_plus_one=False,
        )
        return {
            "roi_score_nms": nms_outputs[0],
            "roi_bbox_nms": nms_outputs[1],
            "roi_class_nms": nms_outputs[2],
            "roi_batch_splits_nms": nms_outputs[3],
            "roi_keeps_nms": nms_outputs[4],
            "roi_keeps_size_nms": nms_outputs[5],
        }

    def get_input_names(self):
        return ["class_prob", "roi_pred_bbox", "roi_batch_splits"]

    def get_output_names(self):
        return ["roi_score_nms", "roi_bbox_nms", "roi_class_nms",
                "roi_batch_splits_nms", "roi_keeps_nms", "roi_keeps_size_nms"]


if __name__ == '__main__':
    logger = setup_logger()
    logger.info("example: BoxWithNMSLimit")

    m = BoxWithNMSLimit()

    data = get_inputs("class_prob", "roi_pred_bbox", "roi_batch_splits",
                      root="/autox-sz/users/dongqixu/share/trt_plugins/BoxWithNMSLimit", map_location="cpu")
    export_onnx(m, data, "model.onnx")
    targets = m(data)

    TensorRTModel.build_engine("model.onnx", "model.trt", 4, device="CUDA")
    e = TensorRTModel("model.trt")
    outputs = e.inference(data)

    # get sort indices for nms
    _, indices = targets["roi_score_nms"].sort(descending=True)

    # compare torch output and tensorrt output
    assert len(targets) == len(outputs), "Number of outputs does not match!"
    targets = [(k, v.cuda()) for k, v in targets.items()]
    for i, (name, tensor) in enumerate(targets):
        if name in {"roi_keeps_nms", "roi_keeps_size_nms"}:
            continue
        logger.info(name)
        if tensor.size(0) == indices.size(0):
            tensor = tensor[indices]
        diff = outputs[name] - tensor
        unique = torch.unique(diff)
        logger.info("unique\n{}".format(unique))
        logger.info("max\n{}".format(torch.abs(unique).max()))
        assert torch.abs(unique).max() < 1e-3
