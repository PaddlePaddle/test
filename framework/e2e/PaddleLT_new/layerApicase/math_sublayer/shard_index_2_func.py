import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: shard_index_2
    api简介: 通过对Tensor中的单个值或切片应用稀疏加法，从而得到输出的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.shard_index(input,  index_num=20, nshards=4, shard_id=1, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(2, 17, [4, 2, 1]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(2, 17, [4, 2, 1]).astype('int32'), )
    return inputs

