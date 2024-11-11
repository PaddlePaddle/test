import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: repeat_interleave_0
    api简介: 沿着指定轴 axis 对输入 x 进行复制，创建并返回到一个新的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.repeat_interleave(x,  repeats=3, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(4,), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([4]).astype('float32'), )
    return inputs

