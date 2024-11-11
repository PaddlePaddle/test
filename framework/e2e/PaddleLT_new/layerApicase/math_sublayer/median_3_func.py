import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: median_3
    api简介: 沿给定的轴 axis 计算 x 中元素的中位数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.median(x,  axis=0, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(3, 4), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[2, 3, 1, 1], [10, 1, 15, 'nan'], [4, 8, 'nan', 7]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[2, 3, 1, 1], [10, 1, 15, 'nan'], [4, 8, 'nan', 7]]).astype('float32'), )
    return inputs

