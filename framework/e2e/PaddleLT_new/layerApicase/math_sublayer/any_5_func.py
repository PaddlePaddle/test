import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: any_5
    api简介: 对指定维度上的Tensor元素进行逻辑或运算，并输出相应的计算结果
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.any(x,  axis=-1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.bool, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 2, [2, 4, 4, 2]).astype('bool'), dtype='bool', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 2, [2, 4, 4, 2]).astype('bool'), )
    return inputs

