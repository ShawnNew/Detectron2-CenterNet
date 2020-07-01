
This directory contains:

1. A script that converts a detectron2 model to caffe2 format.

2. An example that loads a Mask R-CNN model in caffe2 format and runs inference.

See [tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
for their usage.

## Benchmark

**retinanet_R_50_FPN_1x**

|    Format    | Batch size | Inf time (s) | box AP  | Engine time (ms) |
| :----------: | :--------: | :----------: | :-----: | :--------------: |
| Model zoo    | 1          | 0.067458     | 37.4241 | -                |
| Traceable    | 1          | 0.077649     | 37.3129 | -                |
| TensorRT     | 1          | 0.066281     | 37.3129 | 30.9104          |
| FP16         | 1          | 0.042832     | 37.3293 | 8.8474           |
| INT8         | 1          | 0.038972     | 35.7078 | 5.4228           |
| FP16 + INT8  | 1          | 0.039010     | 35.6859 | 5.2963           |
| Model zoo    | 4          | 0.341447     | 37.4342 | -                |
| Traceable    | 4          | 0.296000     | 37.3129 | -                |
| TensorRT     | 4          | 0.256557     | 37.3136 | 124.0081         |
| FP16         | 4          | 0.159615     | 37.3257 | 28.0756          |
| INT8         | 4          | 0.145380     | 35.6725 | 16.2445          |
| FP16 + INT8  | 4          | 0.145969     | 35.6526 | 15.7442          |

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

|    Format    | Batch size | Inf time (s) | box AP  | mask AP | Engine time (ms) |
| :----------: | :--------: | :----------: | :-----: | :-----: | :--------------: |
| Model zoo    | 1          | 0.047004     | 38.6500 | 35.2411 | -                |
| Traceable    | 1          | 0.063912     | 38.5594 | 35.1413 | -                |
| Model zoo    | 4          | 0.219711     | 38.6539 | 35.2287 | -                |
| Traceable    | 4          | 0.223492     | 34.2371 | 31.4533 | -                |

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

|    Format    | Batch size | Inf time (s) | mIoU    | box AP  | mask AP | PQ      | Engine time (ms) |
| :----------: | :--------: | :----------: | :-----: | :-----: | :-----: | :-----: | :--------------: |
| Model zoo    | 1          | 0.056834     | 41.2219 | 37.6452 | 34.6602 | 39.3896 | -                |
| Model zoo    | 4          | 0.264150     | 41.0528 | 37.6658 | 34.6653 | 39.2604 | -                |

```shell script
./tools/train_net.py --num-gpus 1 \
  --config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml \
  --eval-only MODEL.WEIGHTS /autox/users/dongqixu/envs/model_zoo/panoptic_fpn_R_50_1x.pkl TEST.BATCH_SIZE 4
```
