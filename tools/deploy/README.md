
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

**rpn_R_50_FPN_1x**

ResizeShortestEdge, 45.720, 57.995
ResizeShortestEdge, BATCH_SIZE 4, 45.714, 58.031
ResizeLetterBox, 45.863, 57.892
Traceable, 45.874, 57.902
Traceable, BATCH_SIZE 4, 45.874, 57.902

```shell script
# rpn_R_50_FPN_1x
./tools/train_net.py --num-gpus 1 \
  --config-file configs/COCO-Detection/rpn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /autox-sz/users/dongqixu/share/model_zoo/detectron2/coco-detection/rpn_R_50_FPN_1x.pkl TEST.BATCH_SIZE 4

# possible format options: onnx and tensorrt
./caffe2_converter.py --format onnx --output ./rpn --run-eval \
    --config-file ../../configs/COCO-Detection/rpn_R_50_FPN_1x.yaml \
    MODEL.WEIGHTS /autox-sz/users/dongqixu/share/model_zoo/detectron2/coco-detection/rpn_R_50_FPN_1x.pkl INPUT.DYNAMIC False TEST.BATCH_SIZE 4
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

**resnet50**

|    Format    | Batch size | Inf time (s) | Top1   | Top5   | Engine time (ms) |
| :----------: | :--------: | :----------: | :----: | :----: | :--------------: |
| torchvision  | 64         | 0.1191       | 76.130 | 92.862 | -                |
| TensorRT     | 64         | 0.1179       | 76.130 | 92.862 | 40.4424          |
| FP16         | 64         | 0.1166       | 76.106 | 92.866 | 12.0521          |
| INT8         | 64         | 0.1173       | 76.084 | 92.918 | 6.2329           |
| FP16 + INT8  | 64         | 0.1195       | 76.080 | 92.916 | 6.2211           |
| torchvision  | 1024       | 1.9038       | 76.130 | 92.862 | -                |
| TensorRT     | 1024       | 1.9130       | 76.130 | 92.862 | 613.5728         |
| INT8         | 1024       | 1.9263       | 76.038 | 92.864 | 74.9226          |

```shell script
python imagenet.py /autox-sz/users/dongqixu/dataset/ILSVRC2012/val --output ./resnet50 --batch-size 64 \
    --debug --format tensorrt
```
