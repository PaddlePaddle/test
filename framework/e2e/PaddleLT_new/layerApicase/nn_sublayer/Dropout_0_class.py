import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Dropout_0
    api简介: Dropout是一种正则化手段，该算子根据给定的丢弃概率p ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Dropout(p=0.5, axis=1, )

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

