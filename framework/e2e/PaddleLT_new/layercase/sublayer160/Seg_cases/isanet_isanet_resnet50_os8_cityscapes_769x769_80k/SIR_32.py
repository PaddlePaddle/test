# method:__floordiv__||method:__floordiv__||method:__sub__||method:__floordiv__||method:__floordiv__||method:__sub__||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.common.pad||method:__getitem__||method:reshape||method:transpose||method:reshape
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_1,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [1, 512, 97, 97], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [4], dtype: paddle.int32, stop_gradient: True)
        var_4,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_5,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
    ):
        var_6 = var_0.__floordiv__(2)
        var_7 = var_0.__floordiv__(2)
        var_8 = var_0.__sub__(var_7)
        var_9 = var_1.__floordiv__(2)
        var_10 = var_1.__floordiv__(2)
        var_11 = var_1.__sub__(var_10)
        var_12 = paddle.tensor.manipulation.concat([var_6, var_8, var_9, var_11], axis=0)
        var_13 = paddle.nn.functional.common.pad(var_2, var_12)
        var_14 = var_3.__getitem__(1)
        var_15 = var_13.reshape([0, var_14, var_4, 8, var_5, 8])
        var_16 = var_15.transpose([0, 3, 5, 1, 2, 4])
        var_17 = var_16.reshape([-1, 512, var_4, var_5])
        return var_17


def create_tensor_inputs():
    inputs = (
        paddle.to_tensor([1], dtype=paddle.int32),
        paddle.to_tensor([1], dtype=paddle.int32),
        paddle.rand(shape=[1, 512, 97, 97], dtype=paddle.float32),
        paddle.to_tensor([3, 8, 9, 9], dtype=paddle.int32),
        paddle.to_tensor([98], dtype=paddle.int32),
        paddle.to_tensor([98], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.array([1], dtype="int32"),
        np.array([1], dtype="int32"),
        np.random.random(size=[1, 512, 97, 97]).astype("float32"),
        np.array([3, 8, 9, 9], dtype="int32"),
        np.array([98], dtype="int32"),
        np.array([98], dtype="int32"),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({"FLAGS_prim_all": with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
