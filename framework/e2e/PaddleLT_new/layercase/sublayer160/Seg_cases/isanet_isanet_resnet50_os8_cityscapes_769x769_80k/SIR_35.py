# method:__floordiv__||method:__floordiv__||method:__floordiv__||method:__getitem__||method:__add__||method:__floordiv__||method:__getitem__||method:__add__||api:paddle.tensor.manipulation.slice||api:paddle.tensor.manipulation.concat
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 512, 104, 104], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [4], dtype: paddle.int32, stop_gradient: True)
        var_4,  # (shape: [1, 512, 97, 97], dtype: paddle.float32, stop_gradient: False)
    ):
        var_5 = var_1.__floordiv__(2)
        var_6 = var_2.__floordiv__(2)
        var_7 = var_1.__floordiv__(2)
        var_8 = var_3.__getitem__(2)
        var_9 = var_7.__add__(var_8)
        var_10 = var_2.__floordiv__(2)
        var_11 = var_3.__getitem__(3)
        var_12 = var_10.__add__(var_11)
        var_13 = paddle.tensor.manipulation.slice(var_0, axes=[2, 3], starts=[var_5, var_6], ends=[var_9, var_12])
        var_14 = paddle.tensor.manipulation.concat([var_13, var_4], axis=1)
        return var_14


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 104, 104], dtype=paddle.float32),
        paddle.to_tensor([1], dtype=paddle.int32),
        paddle.to_tensor([1], dtype=paddle.int32),
        paddle.to_tensor([8, 8, 97, 97], dtype=paddle.int32),
        paddle.rand(shape=[1, 512, 97, 97], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 104, 104]).astype("float32"),
        np.array([1], dtype="int32"),
        np.array([1], dtype="int32"),
        np.array([8, 8, 97, 97], dtype="int32"),
        np.random.random(size=[1, 512, 97, 97]).astype("float32"),
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
