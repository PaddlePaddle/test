import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Bilinear_0
    api简介: 该层对两个输入执行双线性张量积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Bilinear(in1_features=5, in2_features=4, out_features=8, )

    def forward(self, data0, data1, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, data1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 5), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, 4), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([5, 5]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([5, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([5, 5]).astype('float32'), -1 + (1 - -1) * np.random.random([5, 4]).astype('float32'), )
    return inputs

