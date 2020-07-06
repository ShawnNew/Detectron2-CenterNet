# Notes

* commit: 9ce0300c
* date: 6 July, 2020

## Datasets

* Scale dataset: register_scale()
  * NUM_CLASSES: 9
  * network: faster_rcnn, retinanet

## Config

* INPUT.DYNAMIC = True, default to use ResizeShortestEdge transform
* TEST.BATCH_SIZE = 1, used to set the default inference batch size

## Transform

* ResizeLetterBox, if the input is not dynamic, the image is resized with letter box transform (without any padding)
* ImageList, if the input is not dynamic, the image is padded with zero to fixed size after normalization

## Evaluator

* CityscapesEvaluator: per-class AP
* PascalVOCDetectionEvaluator: per-class AP

## Export

* onnx_tensorrt: fork from [onnx_tensorrt](https://github.com/onnx/onnx-tensorrt), support quantization layers selection and pytorch cuda tensor input/output

* export_onnx: support onnx graph optimization to remove copy between cpu and gpu, call torch.nn.functional.interpolate function with fixed size arguments

* c10: modify `Caffe2RPN`, `Caffe2ROIPooler`, `Caffe2FastRCNNOutputsInference`, `Caffe2MaskRCNNInference` to support onnx model export for `GeneralizedRCNN`

* meta_modeling: define base class `MetaModel` for traceable model for both inference and onnx export, extended by `RetinaNetModel` and `GeneralizedRCNNModel`

* tensorrt: define base class `TensorRTModel` to build and inference TensorRT engines, extended by `TensorRTRetinaNet`

## Deploy interface

* caffe2_converter: support onnx export, onnx traceable model inference, TensorRT engine build and inference
* imagenet: classification example

## LOGGER

* default verbosity is set to logging.INFO
