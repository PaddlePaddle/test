import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: sequence_mask_base
    api简介: 该层根据输入 x 和 maxlen 输出一个掩码
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.sequence_mask(x,  maxlen=10, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 10, [2, 4, 9, 8]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 10, [2, 4, 9, 8]).astype('int32'), )
    return inputs

