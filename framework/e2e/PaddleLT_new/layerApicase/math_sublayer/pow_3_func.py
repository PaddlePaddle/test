import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: pow_3
    api简介: 指数算子，逐元素计算 x 的 y 次幂指数算子，逐元素计算 x 的 y 次幂
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.pow(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([3.0], dtype='float32', stop_gradient=False), paddle.to_tensor([[0.68100125, 0.02188216]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([3.0]).astype('float32'), np.array([[0.68100125, 0.02188216]]).astype('float32'), )
    return inputs

