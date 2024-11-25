import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: BatchNorm1D_8
    api简介: 1维BN批归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.BatchNorm1D(num_features=1, momentum=0.1, epsilon=1e-05, weight_attr=False, bias_attr=False, data_format='NCL', )

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
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype('float32'), )
    return inputs

