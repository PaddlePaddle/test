import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: UpsamplingBilinear2D_4
    api简介: 该OP用于双线性插值插值调整一个batch中图片的大小
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.UpsamplingBilinear2D(size=[10, 10], data_format='NCHW', )

    def forward(self, data0, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([4, 5, 20, 20]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([4, 5, 20, 20]).astype('float32'), )
    return inputs

