import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: isclose_1
    api简介: 逐个检查x和y的所有元素是否均满足∣∣x−y∣∣≤atol+rtol×∣∣y∣∣
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.isclose(x, y,  equal_nan=True, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([10000.0, 1e-07, 'nan', 1.0, 3.0, 0.0], dtype='float32', stop_gradient=False), paddle.to_tensor([10000.01, 1e-06, 'nan', 'nan', 3.0, 'nan'], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([10000.0, 1e-07, 'nan', 1.0, 3.0, 0.0]).astype('float32'), np.array([10000.01, 1e-06, 'nan', 'nan', 3.0, 'nan']).astype('float32'), )
    return inputs

