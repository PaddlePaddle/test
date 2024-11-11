import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: affine_grid_base
    api简介: 生成仿射变换前后的feature maps的坐标映射关系
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, theta, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.affine_grid(theta,  out_shape=[1, 2, 3, 3], align_corners=True, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 2, 3), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (2 - -1) * np.random.random([1, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (2 - -1) * np.random.random([1, 2, 3]).astype('float32'), )
    return inputs

