import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: InstanceNorm1D_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.InstanceNorm1D(
            num_features=2,
            epsilon=0.00001,
            momentum=0.9,
            data_format='NCL',
        )

    def forward(self, data, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 0, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 0, 1]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 0, 1]).astype('float32'), )
    return inputs

