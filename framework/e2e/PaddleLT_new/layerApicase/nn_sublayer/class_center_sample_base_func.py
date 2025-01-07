import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: class_center_sample_base
    api简介: 类别中心采样方法, 从全量的类别中心采样一个子集类别中心参与训练
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, label, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.class_center_sample(label,  num_classes=20, num_samples=6, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 19, [200]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 19, [200]).astype('int32'), )
    return inputs

