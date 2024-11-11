import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: crop_0
    api简介: 根据偏移量（offsets）和形状（shape），裁剪输入（x）Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.crop(x,  shape=[2, 2], offsets=(0, 1), )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('float32'), )
    return inputs

