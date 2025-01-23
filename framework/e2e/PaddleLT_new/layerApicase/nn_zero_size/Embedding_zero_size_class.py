import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Embedding_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Embedding(
            num_embeddings=4,
            embedding_dim=4,
            sparse=False,
        )

    def forward(self, data, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, 0, -1, -1), dtype=paddle.int32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(4 * np.random.random([3, 0, 1, 1]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (4 * np.random.random([3, 0, 1, 1]).astype('int32'), )
    return inputs

