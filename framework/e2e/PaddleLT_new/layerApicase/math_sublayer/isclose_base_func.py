import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: isclose_base
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
        out = paddle.isclose(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]], dtype='float32', stop_gradient=False), paddle.to_tensor([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]]).astype('float32'), np.array([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]]).astype('float32'), )
    return inputs

