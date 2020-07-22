# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p

# from onnx import numpy_helper
import logging
import numpy as np
import onnx
import six
import struct
import tensorrt as trt
from onnx import helper as onnx_helper
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict

from .config import Config
from .tensorrt_engine import Engine

logger = logging.getLogger(__name__)

libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)


def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count


def decode_calibration_cache(cache_f):
    table = {}
    with open(cache_f) as f:
        cache = f.read().splitlines()
        logger.info(cache[0])
        for pair in cache[1:]:
            layer_name, scale = pair.replace(":", "").rsplit(maxsplit=1)
            scale = struct.unpack("!f", bytes.fromhex(scale))[0] * (2 ** 7 - 1)
            # print("{:60} {:12.4f}".format(layer_name, scale))
            table[layer_name] = scale
    return table


_config = Config()

if _config.USE_PYBIND:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if not _config.USE_PYBIND:
    # from . import parser
    # from . import runtime as parser_runtime
    raise NotImplementedError


class TensorRTBackendRep(BackendRep):
    def __init__(self, model, device, max_batch_size=32,
                 max_workspace_size=None, serialize_engine=False, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self._logger = TRT_LOGGER
        self.builder = trt.Builder(self._logger)
        self.network = self.builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self._logger)

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
        if max_workspace_size is None:
            max_workspace_size = 1 << 28

        self.builder.max_batch_size = max_batch_size
        self.builder.max_workspace_size = max_workspace_size

        if "fp16_mode" in kwargs:
            self.builder.fp16_mode = kwargs["fp16_mode"]
            assert not kwargs["fp16_mode"] or self.builder.platform_has_fast_fp16
        if "int8_mode" in kwargs:
            self.builder.int8_mode = kwargs["int8_mode"]
            assert not kwargs["int8_mode"] or self.builder.platform_has_fast_int8
            # perform calibration only when no quantization_layers nor exclude_layers are provided
            if self.builder.int8_mode and kwargs["quantization_layers"] is None \
                    and kwargs["exclude_layers"] is None:
                int8_calibrator = kwargs["int8_calibrator"]
                int8_calibrator.check_input_validity(self.network)
                self.builder.int8_calibrator = int8_calibrator

        logger.info("NetworkDefinition:")
        for layer in self.network:
            input_shape = ["{} {}".format(layer.get_input(i).name, layer.get_input(i).shape)
                           for i in range(layer.num_inputs)]
            output_shape = ["{} {}".format(layer.get_output(i).name, layer.get_output(i).shape)
                            for i in range(layer.num_outputs)]
            print("{:40} : {:30} -> {:30}".format(layer.name, ", ".join(input_shape), ", ".join(output_shape)))
        logger.info("NetworkInput:")
        for i in range(self.network.num_inputs):
            tensor = self.network.get_input(i)
            print(tensor.name, tensor.shape)
        logger.info("NetworkOutput:")
        for i in range(self.network.num_outputs):
            tensor = self.network.get_output(i)
            print(tensor.name, tensor.shape)

        if self.builder.int8_mode and kwargs["quantization_layers"] is not None:
            quantization_layers = kwargs["quantization_layers"]
            table = decode_calibration_cache(kwargs["int8_calibrator"].cache_file)

            for layer in self.network:
                if layer.name in quantization_layers:
                    for i in range(layer.num_inputs):
                        tensor = layer.get_input(i)
                        if tensor.name in table:
                            value = table[tensor.name]
                            tensor.dynamic_range = (-value, value)
                    for i in range(layer.num_outputs):
                        tensor = layer.get_output(i)
                        if tensor.name in table:
                            value = table[tensor.name]
                            tensor.dynamic_range = (-value, value)

        if self.builder.int8_mode and kwargs["exclude_layers"] is not None:
            exclude_tensors = set()
            exclude_layers = kwargs["exclude_layers"]
            table = decode_calibration_cache(kwargs["int8_calibrator"].cache_file)

            # add inputs to excluded tensors
            for layer in self.network:
                if layer.name in exclude_layers:
                    for i in range(layer.num_inputs):
                        tensor = layer.get_input(i)
                        exclude_tensors.add(tensor.name)

            # fill all non-excluded tensors
            for layer in self.network:
                for i in range(layer.num_inputs):
                    tensor = layer.get_input(i)
                    if tensor.name in table and tensor.name not in exclude_tensors:
                        value = table[tensor.name]
                        tensor.dynamic_range = (-value, value)
                for i in range(layer.num_outputs):
                    tensor = layer.get_output(i)
                    if tensor.name in table and tensor.name not in exclude_tensors:
                        value = table[tensor.name]
                        tensor.dynamic_range = (-value, value)

        trt_engine = self.builder.build_cuda_engine(self.network)
        if trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        if serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine)
        self.engine = Engine(trt_engine)
        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type

    def _set_device(self, device):
        self.device = device
        assert (device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def _serialize_deserialize(self, trt_engine):
        self.runtime = trt.Runtime(TRT_LOGGER)
        serialized_engine = trt_engine.serialize()
        del self.parser  # Parser no longer needed for ownership of plugins
        trt_engine = self.runtime.deserialize_cuda_engine(
            serialized_engine)
        return trt_engine

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        outputs = self.engine.run(inputs)
        output_names = [output.name for output in self.engine.outputs]

        for i, (name, array) in enumerate(zip(output_names, outputs)):
            output_shape = self._output_shapes[name]
            # HACK WAR for unknown output shape in run_node
            if output_shape == (-99,):
                # WAR for TRT requiring at least 2 dims (NC)
                min_dims = 2
                if _tensorrt_version()[0] < 4:
                    # WAR for TRT only supporting 4D (NCHW) tensors
                    min_dims = 4
                if array.ndim == min_dims:
                    npadding_dims = count_trailing_ones(array.shape)
                    if npadding_dims > 0:
                        outputs[i] = array.reshape(
                            array.shape[:-npadding_dims])
            else:
                # HACK WAR replace fixed batch dim with variable
                if self._output_dtype[name] == onnx.TensorProto.INT64 and array.dtype == np.int32:
                    casted_output = np.array(outputs[i], dtype=np.int64)
                    if np.equal(outputs[i], casted_output).all():
                        outputs[i] = np.array(outputs[i], dtype=np.int64)

        outputs_tuple = namedtupledict('Outputs', output_names)(*outputs)
        return namedtupledict('Outputs', output_names)(*outputs)


def np2onnx_dtype(np_dtype):
    if np_dtype == np.dtype('float32'):
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.dtype('float16'):
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.dtype('int64'):
        return onnx.TensorProto.INT64
    elif np_dtype == np.dtype('int32'):
        return onnx.TensorProto.INT32
    elif np_dtype == np.dtype('int8'):
        return onnx.TensorProto.INT8
    else:
        raise TypeError("Unsupported data type:", np_dtype)


def make_node_test_model(node, inputs, use_weights=True):
    # HACK TODO: The output info is unknown here; not sure what the best solution is
    output_dtype = np.float32  # Dummy value only
    output_shape = [-99]  # Dummy value only
    graph_inputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(array.dtype), array.shape)
        for name, array in zip(node.input, inputs)]
    graph_outputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(output_dtype), output_shape)
        for name in node.output]
    if use_weights:
        # Add initializers for all inputs except the first
        initializers = [onnx_helper.make_tensor(
            name, np2onnx_dtype(array.dtype), array.shape, array.flatten().tolist())
            for name, array in zip(node.input[1:], inputs[1:])]
    else:
        initializers = []
    graph = onnx_helper.make_graph(
        [node], "RunNodeGraph_" + node.op_type,
        graph_inputs, graph_outputs, initializer=initializers)
    model = onnx_helper.make_model(graph)
    return model


class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        """
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, device, **kwargs)

    @classmethod
    def run_model(cls, model, inputs, device='CUDA:0', **kwargs):
        """Build and run an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        return cls.prepare(model, device, **kwargs).run(inputs)

    @classmethod
    def run_node(cls, node, inputs, device='CUDA:0'):
        """Build and run an engine from the given node.
        node -- An ONNX node as a deserialized protobuf.
        Note: This function is intended for testing purposes only;
              use prepare() or run_model() for other purposes.
        """
        super(TensorRTBackend, cls).run_node(node, inputs, device)
        # HACK TODO: This is somewhat dodgy. We first try with weights for all
        #            inputs but the first, then we try again with no weights if
        #            the first try fails.
        model = make_node_test_model(node, inputs, use_weights=True)
        try:
            results = TensorRTBackend.prepare(model, device).run(inputs[:1])
        except RuntimeError:
            model = make_node_test_model(node, inputs, use_weights=False)
            results = TensorRTBackend.prepare(model, device).run(inputs)
        return results

    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        return device.type == DeviceType.CUDA


prepare = TensorRTBackend.prepare
run_node = TensorRTBackend.run_node
run_model = TensorRTBackend.run_model
supports_device = TensorRTBackend.supports_device