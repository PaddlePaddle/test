# method:__add__||api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[21],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[21, 2048, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[21],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[21],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[21, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[21, 512, 1, 1],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [1, 512, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,    # (shape: [1, 2048, 64, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = var_0.__add__(var_1)
        var_4 = paddle.nn.functional.common.dropout2d(var_3, p=0.1, training=self.training, data_format='NCHW', name=None)
        var_5 = paddle.nn.functional.conv._conv_nd(var_4, self.parameter_5, bias=self.parameter_0, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_6 = paddle.nn.functional.common.dropout2d(var_1, p=0.1, training=self.training, data_format='NCHW', name=None)
        var_7 = paddle.nn.functional.conv._conv_nd(var_6, self.parameter_4, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_8 = paddle.nn.functional.common.dropout2d(var_0, p=0.1, training=self.training, data_format='NCHW', name=None)
        var_9 = paddle.nn.functional.conv._conv_nd(var_8, self.parameter_4, bias=self.parameter_3, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_10 = paddle.nn.functional.common.dropout2d(var_2, p=0.1, training=self.training, data_format='NCHW', name=None)
        var_11 = paddle.nn.functional.conv._conv_nd(var_10, self.parameter_1, bias=self.parameter_2, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        return var_5, var_7, var_9, var_11


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 64, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 512, 64, 64], dtype=paddle.float32),
        paddle.rand(shape=[1, 2048, 64, 64], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 64, 64]).astype('float32'),
        np.random.random(size=[1, 512, 64, 64]).astype('float32'),
        np.random.random(size=[1, 2048, 64, 64]).astype('float32'),
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