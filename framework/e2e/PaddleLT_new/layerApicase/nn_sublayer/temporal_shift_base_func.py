import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: temporal_shift_base
    api简介: 对输入X做时序通道T上的位移操作，为TSM(Temporal Shift Module)中使用的操作
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.temporal_shift(x,  seg_num=3, shift_ratio=0.25, )
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
    inputs = (paddle.to_tensor(0 + (10 - 0) * np.random.random([6, 4, 9, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (10 - 0) * np.random.random([6, 4, 9, 8]).astype('float32'), )
    return inputs

