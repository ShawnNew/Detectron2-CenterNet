#!/usr/bin/env bash
set -e
set -x

: \
  "${batch_size:=4}" \
  "${config:=configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml}" \
  "${weights:=/autox/users/dongqixu/envs/model_zoo/retinanet_R_50_FPN_1x.pkl}" \
  "${network:=retinanet}"

output="$(pwd)/output/${network}_batch_${batch_size}"

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
