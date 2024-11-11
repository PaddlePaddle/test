import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: cosine_similarity_base
    api简介: 计算x1与x2沿axis维度的余弦相似度
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x1, x2, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.cosine_similarity(x1, x2,  axis=1, eps=1e-08, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(2, 3, 8, 8), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(2, 3, 8, 8), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

