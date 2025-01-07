import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Transformer_5
    api简介: Transformer模型
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Transformer(d_model=128, nhead=2, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, )

    def forward(self, data0, data1, data2, data3, data4, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, data1, data2, data3, data4, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 4, 128]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 6, 128]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 6, 6]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 6, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 4, 128]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 6, 128]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 2, 4, 4]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 2, 6, 6]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 2, 6, 4]).astype('float32'), )
    return inputs

