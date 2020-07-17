#!/usr/bin/env bash
set -e
set -x

: \
  "${data:=/autox/users/dongqixu/ILSVRC2012/val}" \
  "${network:=resnet50}" \
  "${batch_size:=64}" \
  "${calibration_batch:=128}"

output="$(pwd)/output/${network}_batch_${batch_size}"

mkdir -p "${output}"

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format onnx | tee "${output}"/onnx.txt

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format torch | tee "${output}"/torch.txt

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format tensorrt | tee "${output}"/tensorrt.txt

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format tensorrt --fp16 | tee "${output}"/tensorrt_fp16.txt

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format tensorrt --int8 \
    --calibration-batch ${calibration_batch} | tee "${output}"/tensorrt_int8.txt

python tools/deploy/imagenet.py ${data} --output "${output}" --batch-size ${batch_size} \
    --debug --format tensorrt --fp16 --int8 \
    --calibration-batch ${calibration_batch} | tee "${output}"/tensorrt_fp16_int8.txt
