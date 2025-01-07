# api:paddle.nn.functional.conv._conv_nd||method:flatten||method:transpose||api:paddle.nn.functional.norm.layer_norm||method:transpose||method:reshape||method:flatten||method:transpose||api:paddle.nn.functional.common.dropout
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[96],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[96, 3, 4, 4],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 3, 544, 640], dtype: paddle.float32, stop_gradient: True)
    ):
        paddle.seed(33)
        var_1 = paddle.nn.functional.conv._conv_nd(var_0, self.parameter_3, bias=self.parameter_2, stride=[4, 4], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        var_2 = var_1.flatten(2)
        var_3 = var_2.transpose([0, 2, 1])
        var_4 = paddle.nn.functional.norm.layer_norm(var_3, normalized_shape=[96], weight=self.parameter_1, bias=self.parameter_0, epsilon=1e-05)
        var_5 = var_4.transpose([0, 2, 1])
        var_6 = var_5.reshape([-1, 96, 136, 160])
        var_7 = var_6.flatten(2)
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.nn.functional.common.dropout(var_8, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        return var_9



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 3, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 3, 544, 640], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 3, 544, 640]).astype('float32'),
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
        paddle.seed(33)
        outs = net(*self.inputs)
        return outs
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()