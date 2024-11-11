import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: allclose_9
    api简介: 逐个检查x和y的所有元素是否均满足∣x−y∣∣≤atol+rtol×∣∣y∣∣
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.allclose(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([], dtype='float32', stop_gradient=False), paddle.to_tensor([], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([]).astype('float32'), np.array([]).astype('float32'), )
    return inputs

