import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Unfold_base
    api简介: 通被称作为im2col过程. 对于每一个输入形状为[N, C, H, W]的 x ，都将计算出一个形状为[N, Cout, Lout]的输出
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Unfold(kernel_sizes=3, )

    def forward(self, data, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 10, 10]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 10, 10]).astype('float32'), )
    return inputs

