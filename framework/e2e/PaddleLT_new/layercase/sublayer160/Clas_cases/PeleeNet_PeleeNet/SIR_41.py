# api:paddle.nn.functional.pooling.avg_pool2d||method:flatten||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.common.linear
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[704, 1000],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1000],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [43, 704, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.avg_pool2d(var_0, kernel_size=[7, 7])
        var_2 = var_1.flatten(1)
        var_3 = paddle.nn.functional.common.dropout(var_2, p=0.05, training=self.training)
        var_4 = paddle.nn.functional.common.linear(x=var_3, weight=self.parameter_0, bias=self.parameter_1, name=None)
        return var_4


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[43, 704, 7, 7], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[43, 704, 7, 7]).astype('float32'),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()
    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
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


if __name__ == '__main__':
    unittest.main()