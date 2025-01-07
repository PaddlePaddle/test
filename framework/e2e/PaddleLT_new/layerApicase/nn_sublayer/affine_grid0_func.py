import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: affine_grid0
    api简介: 生成仿射变换前后的feature maps的坐标映射关系
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.affine_grid( theta=paddle.to_tensor(-1 + (2 - -1) * np.random.random([1, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), out_shape=paddle.to_tensor([1, 2, 3, 3], dtype='int32', stop_gradient=False), align_corners=True, )
        return out



def create_inputspec(): 
    inputspec = ( 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs

