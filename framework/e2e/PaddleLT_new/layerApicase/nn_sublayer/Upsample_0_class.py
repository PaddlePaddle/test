import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Upsample_0
    api简介: 该OP用于插值调整一个batch中2D-或3D-image的大小
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Upsample(size=[12, 12], mode='nearest', align_corners=False, align_mode=0, data_format='NCHW', )

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
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.int64, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 5, 4]).astype('int64'), dtype='int64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-1, 1, [2, 3, 5, 4]).astype('int64'), )
    return inputs

