import tvm
import numpy as np


def test_resnet50():
    in_channel = 3
    out_channel = 32
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [4, 3, 32, 32]
    if not tvm.module.enabled("sophon"):
        print("skip because sophon is not enabled...")
        return
    # if not tvm.get_global_func("tvm.contrib.cudnn.conv2d.output_shape", True):
    #     print("skip because cudnn is not enabled...")
    #     return
    wshape = [in_channel, out_channel, filter_h, filter_w]

    def verify():
        ctx = tvm.sophon(0)
        # f = tvm.build(s, [X, W, Y], "cuda", target_host="llvm", name="conv2d")

        # create sophon module from original data: cmdbuf, prototxt, weight
        # store sophon module to binrary
        # load sophon module from bnrary

        x_data = np.random.uniform(-1, 1, xshape).astype(np.float32);
        w_data = np.random.uniform(-1, 1, wshape).astype(np.float32);
        x = tvm.nd.array(x_data, ctx)
        w = tvm.nd.array(w_data, ctx)
        # y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32),
        #                  ctx)
        # run sophon module
        # f(x, w, y)
        np.testing.assert_allclose(x_data, x.asnumpy())

    verify()


if __name__ == "__main__":
    test_resnet50()
