import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: floor_divifloor_divide_zero_size_funcde_3
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.floor_divide(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-10, 20, [12, 0, 10, 10]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(1, 5, [12, 0, 10, 10]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-10, 20, [12, 0, 10, 10]).astype('int32'), np.random.randint(1, 5, [12, 0, 10, 10]).astype('int32'), )
    return inputs

