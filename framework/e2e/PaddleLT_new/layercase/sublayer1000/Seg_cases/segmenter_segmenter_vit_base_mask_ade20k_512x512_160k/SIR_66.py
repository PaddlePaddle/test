# api:paddle.nn.functional.norm.layer_norm||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||api:paddle.nn.functional.common.linear||method:__floordiv__||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:__getitem__||method:transpose||method:matmul||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.common.dropout||method:matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[768, 768],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[768, 2304],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[768],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 1174, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        paddle.seed(33)
        var_1 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[768], weight=self.parameter_3, bias=self.parameter_4, epsilon=1e-05)
        var_2 = var_1.shape
        var_3 = var_2.__getitem__(1)
        var_4 = var_2.__getitem__(2)
        var_5 = paddle.nn.functional.common.linear(x=var_1, weight=self.parameter_2, bias=None, name=None)
        var_6 = var_4.__floordiv__(12)
        var_7 = var_5.reshape((-1, var_3, 3, 12, var_6,))
        var_8 = var_7.transpose((2, 0, 3, 1, 4,))
        var_9 = var_8.__getitem__(0)
        var_10 = var_8.__getitem__(1)
        var_11 = var_8.__getitem__(2)
        var_12 = var_10.transpose((0, 1, 3, 2,))
        var_13 = var_9.matmul(var_12)
        var_14 = var_13.__mul__(0.125)
        var_15 = paddle.nn.functional.activation.softmax(var_14, axis=-1)
        var_16 = paddle.nn.functional.common.dropout(var_15, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_17 = var_16.matmul(var_11)
        var_18 = var_17.transpose((0, 2, 1, 3,))
        var_19 = var_18.reshape((-1, var_3, var_4,))
        var_20 = paddle.nn.functional.common.linear(x=var_19, weight=self.parameter_1, bias=self.parameter_0, name=None)
        var_21 = paddle.nn.functional.common.dropout(var_20, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        return var_21



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, 768), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 1174, 768], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 1174, 768]).astype('float32'),
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