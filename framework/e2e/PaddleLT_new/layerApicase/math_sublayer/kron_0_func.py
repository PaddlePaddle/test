import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: kron_0
    api简介: 计算两个张量的克罗内克积, 结果是一个合成的张量, 由第二个张量经过第一个张量中的元素缩放 后的组块构成。
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.kron(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.int32, stop_gradient=True), 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-10, 10, [2, 2]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-10, 10, [3, 3, 2]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-10, 10, [2, 2]).astype('int32'), np.random.randint(-10, 10, [3, 3, 2]).astype('int32'), )
    return inputs

