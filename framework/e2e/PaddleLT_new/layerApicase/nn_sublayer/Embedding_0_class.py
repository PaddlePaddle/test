import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Embedding_0
    api简介: 嵌入层(Embedding Layer)
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Embedding(num_embeddings=10, embedding_dim=3, sparse=True, )

    def forward(self, data0, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 10, [3, 1]).astype('int64'), dtype='int64', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 10, [3, 1]).astype('int64'), )
    return inputs

