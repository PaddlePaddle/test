import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: index_sample_base
    api简介: 对输入 x 中的元素进行批量抽样，取 index 指定的对应下标的元素，按index中出现的先后顺序组织，填充为一个新的张量
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, index, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.index_sample(x, index,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 5]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(np.random.randint(0, 3, [4, 5]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 5]).astype('float32'), np.random.randint(0, 3, [4, 5]).astype('int32'), )
    return inputs

