# api:paddle.nn.functional.norm.layer_norm||api:paddle.tensor.attribute.shape||method:__getitem__||method:__getitem__||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:transpose||method:__matmul__||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.common.dropout||method:__matmul__||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
           shape=[1024],
           dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
           shape=[512, 1024],
           dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
           shape=[512, 512],
           dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
           shape=[512],
           dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
           shape=[512, 512],
           dtype=paddle.float32,
        )
    def forward(
        self,
        var_0,    # (shape: [1, 512, 512], dtype: paddle.float32, stop_gradient: False)
    ):
        paddle.seed(33)
        var_1 = paddle.nn.functional.norm.layer_norm(var_0, normalized_shape=[512], weight=self.parameter_3, bias=self.parameter_5, epsilon=1e-06)
        var_2 = var_1.shape
        var_3 = var_2.__getitem__(0)
        var_4 = var_2.__getitem__(1)
        var_5 = paddle.nn.functional.common.linear(x=var_1, weight=self.parameter_4, bias=self.parameter_0, name=None)
        var_6 = var_5.reshape([var_3, var_4, 8, 64])
        var_7 = var_6.transpose([0, 2, 1, 3])
        var_8 = paddle.nn.functional.common.linear(x=var_1, weight=self.parameter_2, bias=self.parameter_1, name=None)
        var_9 = var_8.reshape([var_3, -1, 2, 8, 64])
        var_10 = var_9.transpose([2, 0, 3, 1, 4])
        var_11 = var_10.__getitem__(0)
        var_12 = var_10.__getitem__(1)
        var_13 = var_11.transpose([0, 1, 3, 2])
        var_14 = var_7.__matmul__(var_13)
        var_15 = var_14.__mul__(0.125)
        var_16 = paddle.nn.functional.activation.softmax(var_15, axis=-1)
        var_17 = paddle.nn.functional.common.dropout(var_16, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        var_18 = var_17.__matmul__(var_12)
        var_19 = var_18.transpose([0, 2, 1, 3])
        var_20 = var_19.reshape([var_3, var_4, 512])
        var_21 = paddle.nn.functional.common.linear(x=var_20, weight=self.parameter_7, bias=self.parameter_6, name=None)
        var_22 = paddle.nn.functional.common.dropout(var_21, p=0.0, axis=None, training=self.training, mode='upscale_in_train', name=None)
        return var_22



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, 512), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 512], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1, 512, 512]).astype('float32'),
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