# api:paddle.nn.functional.norm.layer_norm||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:transpose||method:reshape||api:paddle.nn.functional.conv._conv_nd||method:reshape||method:transpose||api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:transpose||method:__matmul__||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.common.dropout||method:__matmul__||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[128, 256],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
           shape=[128, 128],
           dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
           shape=[256],
           dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
           shape=[128, 128, 4, 4],
           dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
           shape=[128, 128],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 16384, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
        var_2,    # (shape: [], dtype: paddle.int32, stop_gradient: True)
    ):
        var_3 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[128], weight=self.parameter_1, bias=self.parameter_3, epsilon=1e-06)
        var_4 = var_3.shape
        var_5 = var_4.__getitem__(0)
        var_6 = var_4.__getitem__(1)
        var_7 = paddle.nn.functional.common.linear(x=var_3, weight=self.parameter_11, bias=self.parameter_5, name=None)
        var_8 = var_7.reshape([var_5, var_6, 2, 64])
        var_9 = var_8.transpose([0, 2, 1, 3])
        var_10 = var_3.transpose([0, 2, 1])
        var_11 = var_10.reshape([var_5, 128, var_1, var_2])
        var_12 = paddle.nn.functional.conv._conv_nd(var_11, self.parameter_10, bias=self.parameter_0, stride=[4, 4], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_13 = var_12.reshape([var_5, 128, -1])
        var_14 = var_13.transpose([0, 2, 1])
        var_15 = paddle.nn.functional.norm.layer_norm(var_14, normalized_shape=[128], weight=self.parameter_2, bias=self.parameter_7, epsilon=1e-05)
        var_16 = paddle.nn.functional.common.linear(x=var_15, weight=self.parameter_4, bias=self.parameter_9, name=None)
        var_17 = var_16.reshape([var_5, -1, 2, 2, 64])
        var_18 = var_17.transpose([2, 0, 3, 1, 4])
        var_19 = var_18.__getitem__(0)
        var_20 = var_18.__getitem__(1)
        var_21 = var_19.transpose([0, 1, 3, 2])
        var_22 = var_9.__matmul__(var_21)
        var_23 = var_22.__mul__(0.125)
        var_24 = paddle.nn.functional.activation.softmax(var_23, axis=-1)
        var_25 = paddle.nn.functional.common.dropout(var_24, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_26 = var_25.__matmul__(var_20)
        var_27 = var_26.transpose([0, 2, 1, 3])
        var_28 = var_27.reshape([var_5, var_6, 128])
        var_29 = paddle.nn.functional.common.linear(x=var_28, weight=self.parameter_8, bias=self.parameter_6, name=None)
        var_30 = paddle.nn.functional.common.dropout(var_29, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        return var_30


def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 16384, 128], dtype=paddle.float32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 16384, 128]).astype('float32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
        np.random.randint(low=0, high=10, size=[1], dtype='int32'),
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