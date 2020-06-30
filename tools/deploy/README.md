
This directory contains:

1. A script that converts a detectron2 model to caffe2 format.

2. An example that loads a Mask R-CNN model in caffe2 format and runs inference.

See [tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for their usage.

## Benchmark

**retinanet_R_50_FPN_1x**

* model zoo: 37.4241 (37.4342), 0.067932 (0.341894)
* traceable model: 37.3129, 0.077887 (0.295680)
* TensorRT: 37.3129 (37.3136), 0.066484 (0.258170), 30.8540 ms (124.0023 ms)

Detection mAP from model zoo is slightly higher with batch size 4.

```shell script
# retinanet_R_50_FPN_1x
./tools/train_net.py --num-gpus 1 \
  --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/retinanet_R_50_FPN_1x.pkl TEST.BATCH_SIZE 4

# possible format options: onnx and tensorrt
cd tools/deploy
./caffe2_converter.py --format tensorrt --output ./retinanet --run-eval \
    --config-file ../../configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml \
    MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/retinanet_R_50_FPN_1x.pkl INPUT.DYNAMIC False TEST.BATCH_SIZE 4
```

**mask_rcnn_R_50_FPN_1x**

* model zoo: 38.6500 (38.6539), 35.2411 (35.2287), 0.047125 (0.219406)
* traceable model: 38.5594 (34.2371), 35.1413 (31.4533), 0.064155 (0.222919)

Performance of traceable model drops when batch size is larger than 1 due to the incorrect implementation of caffe2.

```shell script
# mask_rcnn_R_50_FPN_1x
./tools/train_net.py --num-gpus 1 \
  --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/mask_rcnn_R_50_FPN_1x.pkl TEST.BATCH_SIZE 4

# possible format options: onnx and tensorrt
cd tools/deploy
./caffe2_converter.py --format onnx --output ./mask_rcnn --run-eval \
    --config-file ../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/mask_rcnn_R_50_FPN_1x.pkl INPUT.DYNAMIC False TEST.BATCH_SIZE 4
```

**panoptic_fpn_R_50_1x**

* model zoo: 41.2219, 37.6452, 34.6602, 39.3896, 0.056850

```shell script
./tools/train_net.py --num-gpus 1 \
  --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml \
  --eval-only MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/panoptic_fpn_R_50_1x.pkl
```
