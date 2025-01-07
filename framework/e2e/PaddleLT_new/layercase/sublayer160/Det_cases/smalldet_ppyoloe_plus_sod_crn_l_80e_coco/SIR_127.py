# api:paddle.nn.functional.common.dropout||method:__add__||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.activation.gelu||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:__add__||api:paddle.nn.functional.norm.layer_norm
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[2048, 1024],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[1024, 2048],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[2048],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 361, 1024], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 361, 1024], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.common.dropout(var_0, p=0.1, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_3 = var_1.__add__(var_2)
        var_4 = paddle.nn.functional.norm.layer_norm(var_3, normalized_shape=[1024], weight=self.parameter_6, bias=self.parameter_7, epsilon=1e-05)
        var_5 = paddle.nn.functional.common.linear(x=var_4, weight=self.parameter_4, bias=self.parameter_5, name=None)
        var_6 = paddle.nn.functional.activation.gelu(var_5)
        var_7 = paddle.nn.functional.common.dropout(var_6, p=0.1, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_8 = paddle.nn.functional.common.linear(x=var_7, weight=self.parameter_1, bias=self.parameter_0, name=None)
        var_9 = paddle.nn.functional.common.dropout(var_8, p=0.1, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_10 = var_4.__add__(var_9)
        var_11 = paddle.nn.functional.norm.layer_norm(var_10, normalized_shape=[1024], weight=self.parameter_2, bias=self.parameter_3, epsilon=1e-05)
        return var_11


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 361, 1024], dtype=paddle.float32),
        paddle.rand(shape=[1, 361, 1024], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 361, 1024]).astype('float32'),
        np.random.random(size=[1, 361, 1024]).astype('float32'),
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