import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: CosineSimilarity_5
    api简介: 比较两个tensor的余弦相似度
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.CosineSimilarity(axis=0, eps=1e-08, )

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
        paddle.static.InputSpec(shape=(5, 1, 3), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(1, 3), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-100 + (100 - -100) * np.random.random([5, 1, 3]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-100 + (100 - -100) * np.random.random([1, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-100 + (100 - -100) * np.random.random([5, 1, 3]).astype('float32'), -100 + (100 - -100) * np.random.random([1, 3]).astype('float32'), )
    return inputs

