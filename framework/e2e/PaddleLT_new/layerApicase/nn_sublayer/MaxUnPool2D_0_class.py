import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: MaxUnPool2D_0
    api简介: 2维最大逆池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.MaxUnPool2D(kernel_size=2, )

    def forward(self, data, indices, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, indices, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 4, 40, 40]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(np.random.randint(0, 20, [2, 4, 40, 40]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 4, 40, 40]).astype('float32'), np.random.randint(0, 20, [2, 4, 40, 40]).astype('int32'), )
    return inputs

