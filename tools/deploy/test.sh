#!/usr/bin/env bash
set -e
set -x

# # Example
# batch_size=4 \
#    config=configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
#    weights=/autox/users/dongqixu/envs/model_zoo/mask_rcnn_R_50_FPN_1x.pkl \
#    network=mask_rcnn ./tools/deploy/test.sh

: \
  "${batch_size:=4}" \
  "${config:=configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml}" \
  "${weights:=/autox/users/dongqixu/envs/model_zoo/retinanet_R_50_FPN_1x.pkl}" \
  "${network:=retinanet}"

output="$(pwd)/output/${network}_batch_${batch_size}"

rm -rfv "${output}"
mkdir -p "${output}"

# model zoo
./tools/train_net.py --num-gpus 1 \
  --config-file ${config} \
  --eval-only MODEL.WEIGHTS ${weights} TEST.BATCH_SIZE ${batch_size} | tee "${output}"/model_zoo.txt

cd tools/deploy || exit
# traceable model
./caffe2_converter.py --format onnx --run-eval --debug \
    --output "${output}" \
    --config-file ../../${config} \
    MODEL.WEIGHTS ${weights} INPUT.DYNAMIC False TEST.BATCH_SIZE ${batch_size} | tee "${output}"/traceable_model.txt

# tensorrt
./caffe2_converter.py --format tensorrt --run-eval --debug \
    --output "${output}" \
    --config-file ../../${config} \
    MODEL.WEIGHTS ${weights} INPUT.DYNAMIC False TEST.BATCH_SIZE ${batch_size} | tee "${output}"/tensorrt.txt

./caffe2_converter.py --format tensorrt --run-eval --debug --fp16 \
    --output "${output}" \
    --config-file ../../${config} \
    MODEL.WEIGHTS ${weights} INPUT.DYNAMIC False TEST.BATCH_SIZE ${batch_size} | tee "${output}"/tensorrt_fp16.txt

./caffe2_converter.py --format tensorrt --run-eval --debug --int8 \
    --output "${output}" \
    --config-file ../../${config} \
    MODEL.WEIGHTS ${weights} INPUT.DYNAMIC False TEST.BATCH_SIZE ${batch_size} | tee "${output}"/tensorrt_int8.txt

./caffe2_converter.py --format tensorrt --run-eval --debug --fp16 --int8 --cache \
    --output "${output}" \
    --config-file ../../${config} \
    MODEL.WEIGHTS ${weights} INPUT.DYNAMIC False TEST.BATCH_SIZE ${batch_size} | tee "${output}"/tensorrt_fp16_int8.txt
