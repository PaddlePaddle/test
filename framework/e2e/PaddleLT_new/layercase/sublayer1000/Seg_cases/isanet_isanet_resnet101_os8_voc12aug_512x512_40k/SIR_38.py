# api:paddle.nn.functional.common.dropout2d||api:paddle.nn.functional.conv._conv_nd
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[21, 512, 1, 1],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[21],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 64, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        paddle.seed(33)
        var_1 = paddle.nn.functional.common.dropout2d(var_0, p=0.1, training=self.training, data_format='NCHW', name=None)
        var_2 = paddle.nn.functional.conv._conv_nd(var_1, self.parameter_0, bias=self.parameter_1, stride=[1, 1], padding=[0, 0], padding_algorithm='EXPLICIT', dilation=[1, 1], groups=1, data_format='NCHW', channel_dim=1, op_type='conv2d', use_cudnn=True)
        return var_2



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 512, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 64, 64], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 64, 64]).astype('float32'),
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