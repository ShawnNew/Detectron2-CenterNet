import logging
import os
import re

import pycuda.driver
import tensorrt as trt

from . import to_cuda
from .tensorrt_engine import Holder, torch_to_numpy_dtype

logger = logging.getLogger(__name__)


class PythonEntropyCalibrator(trt.tensorrt.IInt8EntropyCalibrator2):
    def __init__(self, max_calibration_batch, data_loader, preprocess_f, cache_file):
        super(PythonEntropyCalibrator, self).__init__()
        self.max_calibration_batch = max_calibration_batch
        self.data_loader = data_loader
        self.preprocess_f = preprocess_f
        self.cache_file = cache_file

        self.iter = iter(self.data_loader)
        self.index = 0
        self.bindings = []

    def __del__(self):
        del self.iter

    def get_batch(self, names):
        logger.info("Get batch with inputs: {}".format(", ".join(names)))
        try:
            if self.index >= self.max_calibration_batch:
                return None

            batched_inputs = next(self.iter)
            m_inputs = self.preprocess_f(batched_inputs)
            self.index += 1

            # clean previous bindings
            self.bindings = []
            for i, name in enumerate(names):
                name = re.sub("^__", "", name)
                assert name in m_inputs, name
                tensor = to_cuda(m_inputs[name])
                device_buf = pycuda.gpuarray.GPUArray(tensor.shape, dtype=torch_to_numpy_dtype(tensor.dtype),
                                                      gpudata=Holder(tensor))
                logger.info("Input {:5}, name: {:20}, shape: {:20} dtype: {:20}".format(
                    i, name, str(tuple(tensor.shape)), str(tensor.dtype)))
                self.bindings.append(device_buf)

            return [b.ptr for b in self.bindings]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def get_batch_size(self):
        # should return batch 1 because explicit batch size of TensorRT engine.
        return 1

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Read calibration cache from {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        dirname = os.path.dirname(self.cache_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            logger.info("Write calibration cache to {}".format(self.cache_file))
            f.write(cache)
