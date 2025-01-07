import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: gather_tree_base
    api简介: 在整个束搜索(Beam Search)结束后使用
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ids, parents, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.gather_tree(ids, parents,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (paddle.to_tensor([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]], dtype='int32', stop_gradient=False), paddle.to_tensor([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]], dtype='int32', stop_gradient=False), )
    return inputs

