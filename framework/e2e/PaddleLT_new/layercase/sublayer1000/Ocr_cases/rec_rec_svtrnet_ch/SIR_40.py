# api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__mul__||method:__getitem__||method:__getitem__||method:transpose||method:matmul||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.common.dropout||method:matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[128, 384],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[384],
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
           shape=[128, 128],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[128],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [10, 320, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        paddle.seed(33)
        var_1 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[128], weight=self.parameter_2, bias=self.parameter_3, epsilon=1e-06)
        var_2 = paddle.nn.functional.common.linear(x=var_1, weight=self.parameter_0, bias=self.parameter_1, name=None)
        var_3 = var_2.reshape((0, -1, 3, 4, 32,))
        var_4 = var_3.transpose((2, 0, 3, 1, 4,))
        var_5 = var_4.__getitem__(0)
        var_6 = var_5.__mul__(0.1767766952966369)
        var_7 = var_4.__getitem__(1)
        var_8 = var_4.__getitem__(2)
        var_9 = var_7.transpose((0, 1, 3, 2,))
        var_10 = var_6.matmul(var_9)
        var_11 = paddle.nn.functional.activation.softmax(var_10, axis=-1)
        var_12 = paddle.nn.functional.common.dropout(var_11, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_13 = var_12.matmul(var_8)
        var_14 = var_13.transpose((0, 2, 1, 3,))
        var_15 = var_14.reshape((0, -1, 128,))
        var_16 = paddle.nn.functional.common.linear(x=var_15, weight=self.parameter_4, bias=self.parameter_5, name=None)
        var_17 = paddle.nn.functional.common.dropout(var_16, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        return var_17



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, 128), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[10, 320, 128], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[10, 320, 128]).astype('float32'),
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