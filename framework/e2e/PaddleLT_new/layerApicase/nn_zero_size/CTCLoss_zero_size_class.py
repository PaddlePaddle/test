import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: CTCLoss_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, x, y, input_lengths, label_lengths ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(x, y, input_lengths, label_lengths)
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.int32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1), dtype=paddle.int32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1), dtype=paddle.int32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([0, 10, 10]).astype('float32'), dtype='float32', stop_gradient=False), 
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([10, 10]).astype('int32'), dtype='int32', stop_gradient=True), 
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([10]).astype('int64'), dtype='int64', stop_gradient=True), 
        paddle.to_tensor(-1 + (1 - -1) * np.random.random([10]).astype('int64'), dtype='int64', stop_gradient=True),
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -1 + (1 - -1) * np.random.random([0, 10, 10]).astype('float32'), 
        -1 + (1 - -1) * np.random.random([10, 10]).astype('int32'), 
        -1 + (1 - -1) * np.random.random([10]).astype('int64'), 
        -1 + (1 - -1) * np.random.random([10]).astype('int64'), 
    )
    return inputs

