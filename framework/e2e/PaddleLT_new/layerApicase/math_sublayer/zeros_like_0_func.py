import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: zeros_like_0
    api简介: 返回一个和 x 具有相同的形状的全零Tensor，数据类型为 dtype 或者和 x 相同
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.zeros_like(x,  )
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
    inputs = (paddle.to_tensor(np.random.randint(-100, 100, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-100, 100, [2, 3, 4, 4]).astype('int32'), )
    return inputs

