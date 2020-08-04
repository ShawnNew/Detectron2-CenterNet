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

import logging
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver
import pycuda.gpuarray
import tensorrt as trt
import torch
from six import string_types
from pycuda.compyte.dtypes import dtype_to_ctype

from .config import Config

try:
    from trt_profiler import Profiler
except ImportError as error:
    Profiler = None

_config = Config()

logger = logging.getLogger(__name__)

# mapping dict between torch and numpy dtypes
numpy_dtype_mapping = {
    # signed integers
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,

    # unsinged inters
    torch.uint8: np.uint8,

    # floating point
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.float64: np.float64,
    torch.double: np.float64
}


def torch_to_numpy_dtype(dtype):
    if dtype not in numpy_dtype_mapping:
        raise ValueError(f"{dtype} has no PyTorch equivalent")
    else:
        candidate = numpy_dtype_mapping[dtype]
        # we can raise exception early by checking of the type can be used with pycuda. Otherwise
        # we realize it only later when using the array
        try:
            _ = dtype_to_ctype(candidate)
        except ValueError:
            raise ValueError(f"{dtype} cannot be used in pycuda")
        else:
            return candidate


def numpy_to_torch_dtype(dtype):
    for dtype_t, dtype_n in numpy_dtype_mapping.items():
        if dtype_n == dtype:
            return dtype_t


class Holder(pycuda.driver.PointerHolderBase):

    def __init__(self, tensor):
        super().__init__()
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
        self.tensor = tensor
        self.gpu_data = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    def __int__(self):
        return self.gpu_data

    # without an __index__ method, arithmetic calls to the GPUArray backed by this pointer fail
    # not sure why, this needs to return some integer, apparently
    def __index__(self):
        return self.gpu_data


class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)

        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                     trt.DataType.HALF: np.float16,
                     trt.DataType.INT8: np.int8}
        if hasattr(trt.DataType, "INT32"):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)
        self._host_buf = None

        # gpuarray is initialized only if it is a output binding
        if self.is_input:
            self._device_buf = None
            self._tensor = None
            self._shared_array = None
        else:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
            self._tensor = torch.zeros(self.shape, dtype=numpy_to_torch_dtype(self.dtype), device=torch.device("cuda"))
            self._shared_array = pycuda.gpuarray.GPUArray(self._tensor.shape,
                                                          dtype=torch_to_numpy_dtype(self._tensor.dtype),
                                                          gpudata=Holder(self._tensor))

    def fill_(self, tensor):
        # share storage of torch.Tensor and pycuda.gpuarray
        assert self.is_input, "not allow to fill output binding"
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
        self._tensor = tensor
        self._device_buf = pycuda.gpuarray.GPUArray(tensor.shape, dtype=torch_to_numpy_dtype(tensor.dtype),
                                                    gpudata=Holder(tensor))

    def get_async(self, stream):
        if not self.is_input:
            pycuda.driver.memcpy_dtod_async(self._shared_array.gpudata, self._device_buf.gpudata,
                                            self._device_buf.nbytes, stream)
        return self._tensor

    @property
    def ptr(self):
        assert self._device_buf is not None, "device_buf is not initialized!"
        return self._device_buf.ptr

    """
    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf

    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf

    def get_async(self, stream):
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst
    """


def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x


def check_input_validity(input_idx, input_array, input_binding):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()):
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                             (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        # TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                                (input_idx, input_binding.dtype, input_array.dtype))
        elif isinstance(input_array, torch.Tensor):
            mapping = {
                torch.float32: np.float32,
                torch.int32: np.int32,
            }
            if input_array.dtype == torch.int64 and input_binding.dtype == np.int32:
                casted_input_array = input_array.to(torch.int32)
                if torch.equal(input_array, casted_input_array.to(input_array.dtype)):
                    input_array = casted_input_array
                else:
                    raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                                    (input_idx, input_binding.dtype, input_array.dtype))
            if mapping[input_array.dtype] != input_binding.dtype:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                                (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i) for i in range(nbinding)]
        self.inputs = [b for b in bindings if b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]
        self.bindings = bindings

        # logger
        for b in self.inputs:
            logger.info("Input {:6}, name: {:20}, shape: {:20} dtype: {:20}".format(
                b.index, b.name, str(b.shape), str(b.dtype)))
        for b in self.outputs:
            logger.info("Output {:5}, name: {:20}, shape: {:20} dtype: {:20}".format(
                b.index, b.name, str(b.shape), str(b.dtype)))
        logger.info("Shared output tensors:")
        for b in self.outputs:
            logger.info("Tensor {:5}, name: {:20}, shape: {:20}, dtype: {:20}".format(
                b.index, b.name, str(tuple(b._tensor.shape)), str(b._tensor.dtype)))

        # for binding in self.inputs + self.outputs:
        #     _ = binding.device_buffer  # Force buffer allocation
        # for binding in self.outputs:
        #     _ = binding.host_buffer  # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()
        # register TensorRT profiler
        self.count = 0
        self.engine_time = 0
        if Profiler is not None:
            assert callable(Profiler), "Profiler is not callable"
            self.context.profiler = Profiler()
        else:
            # logger.error("Fail to import TensorRT profiler")
            pass

    def __del__(self):
        if self.engine is not None:
            del self.engine

    def run(self, inputs):
        # len(inputs) > len(self.inputs) with Shape operator, input is never used
        # len(inputs) == len(self.inputs) for other operators
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]

        batch_size = 0
        if inputs:
            batch_size = inputs[0].shape[0]
        else:
            batch_size = self.outputs[0].shape[0]

        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            if not input_array.is_cuda:
                input_array = input_array.cuda()
            input_binding.fill_(input_array)
            # input_binding_array = input_binding.device_buffer
            # input_binding_array.set_async(input_array, self.stream)

        self.stream.synchronize()
        start_time = time.perf_counter()
        if Profiler is None:
            self.context.execute_async_v2(
                self.binding_addrs, self.stream.handle)
            self.stream.synchronize()
        else:
            self.context.execute_v2(self.binding_addrs)
        self.engine_time += (time.perf_counter() - start_time) * 1000
        self.count += 1

        results = [output.get_async(self.stream) for output in self.outputs]
        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):
        self.context.execute_async(
            batch_size, self.binding_addrs, self.stream.handle)

    def report_engine_time(self, filename: str, threshold: float):
        if Profiler is not None:
            engine_time = self.context.profiler.report_engine_time(filename, threshold)
        else:
            engine_time = self.engine_time / self.count
        logger.info("Average engine time: {:.4f} ms".format(self.engine_time / self.count))
        return engine_time

    @property
    def binding_addrs(self):
        return [b.ptr for b in self.bindings]
