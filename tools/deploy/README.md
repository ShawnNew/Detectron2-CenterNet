
This directory contains:

1. A script that converts a detectron2 model to caffe2 format.

2. An example that loads a Mask R-CNN model in caffe2 format and runs inference.

See [tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for their usage.

## Benchmark

**retinanet_R_50_FPN_1x**

* model zoo: 37.4241 (37.4342), 0.067417 (0.341997)
* traceable model: 37.3129, 0.076934 (0.295314)
* TensorRT: 37.3136, 0.090066 (0.366513), 30.7647 ms (123.0706 ms)

Detection mAP from model zoo is slightly higher with batch size 4.

```bash
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

* model zoo: 38.6500 (38.6539), 35.2411 (35.2287), 0.046978 (0.220222)
* traceable model: 38.5594 (), 35.1413 (), 0.063897 ()

```bash
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
