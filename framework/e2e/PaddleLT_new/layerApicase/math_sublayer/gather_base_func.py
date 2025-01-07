import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: gather_base
    api简介: 根据索引 index 获取输入 x 的指定 aixs 维度的条目，并将它们拼接在一起
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.gather(x,  index=paddle.to_tensor([2, 0, 1], dtype='int32', stop_gradient=False), )
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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 3, 4, 4]).astype('float32'), )
    return inputs

