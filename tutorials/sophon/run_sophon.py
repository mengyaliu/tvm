"""
Sophon Deployment
===================
"""

import tvm
import numpy as np
import logging


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


######################################################################
# Execute on TVM
# ---------------------------------------------
# load input data to cpu
intput_shape = [1, 3, 224, 224]
input_data = np.fromfile("../../sophon-metainfo/resnet50/data/resnet50_input_data.bin", dtype='float32').reshape(intput_shape)
input = tvm.nd.array(input_data, tvm.cpu(0))

# upload weight data to device
weight_data = np.fromfile("../../sophon-metainfo/resnet50/resnet50_weight.bin", dtype='float32')
weight = tvm.nd.array(weight_data, tvm.sophon(0))

# init output
output_shape = [1, 1000]
output = tvm.nd.array(np.zeros(output_shape, dtype='float32'), tvm.cpu(0))

# load Loadable
resnet50 = tvm.module.load("../../sophon-metainfo/resnet50/resnet50.tvm_meta.json")

# run Loadable
resnet50(input, weight, output)

# run Loadable by function name
# resnet50["__tvm_reset50_1__"](input, weight, output)

# save output data
output.asnumpy().astype("float32").tofile("/tmp/cww.tvm.output.bin")
