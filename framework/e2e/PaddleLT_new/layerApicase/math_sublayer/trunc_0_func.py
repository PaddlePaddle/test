import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: trunc_0
    api简介: 将输入 Tensor 的小数部分置0，返回置0后的 Tensor ，如果输入 Tensor 的数据类型为整数，则不做处理
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.trunc(input,  )
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
    inputs = (paddle.to_tensor(np.random.randint(-10, 10, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-10, 10, [2, 3, 4, 4]).astype('int32'), )
    return inputs

