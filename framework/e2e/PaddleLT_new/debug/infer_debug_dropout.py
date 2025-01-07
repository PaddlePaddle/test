import os
import numpy as np
import paddle
import paddle.inference as paddle_infer


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[2048],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[2048, 1024],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[1024, 2048],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 169, 1024], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 169, 1024], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.common.dropout(
            var_0, p=0.1, axis=None, training=self.training, mode="upscale_in_train", name=None
        )
        var_3 = var_1.__add__(var_2)
        var_4 = paddle.nn.functional.norm.layer_norm(
            var_3, normalized_shape=[1024], weight=self.parameter_5, bias=self.parameter_3, epsilon=1e-05
        )
        var_5 = paddle.nn.functional.common.linear(x=var_4, weight=self.parameter_7, bias=self.parameter_0, name=None)
        var_6 = paddle.nn.functional.activation.gelu(var_5)
        var_7 = paddle.nn.functional.common.dropout(
            var_6, p=0.1, axis=None, training=self.training, mode="upscale_in_train", name=None
        )
        var_8 = paddle.nn.functional.common.linear(x=var_7, weight=self.parameter_6, bias=self.parameter_1, name=None)
        var_9 = paddle.nn.functional.common.dropout(
            var_8, p=0.1, axis=None, training=self.training, mode="upscale_in_train", name=None
        )
        var_10 = var_4.__add__(var_9)
        var_11 = paddle.nn.functional.norm.layer_norm(
            var_10, normalized_shape=[1024], weight=self.parameter_2, bias=self.parameter_4, epsilon=1e-05
        )
        return var_11


seed = 33


def reset(seed):
    """
    重置模型图
    :param seed: 随机种子
    :return:
    """
    paddle.enable_static()
    paddle.disable_static()
    paddle.seed(seed)
    np.random.seed(seed)
    np.set_printoptions(threshold=5, edgeitems=3)


# def _net_input():
#     """get input"""
#     reset(seed)
#     data = (
#         paddle.to_tensor(np.random.random(size=[1, 169, 1024]).astype('float32')),
#         paddle.to_tensor(np.random.random(size=[1, 169, 1024]).astype('float32')),
#     )
#     return data

data = [
    np.random.random(size=[1, 169, 1024]).astype("float32"),
    np.random.random(size=[1, 169, 1024]).astype("float32"),
]

tensor_data = [paddle.to_tensor(data[0]), paddle.to_tensor(data[1])]


def _net_instant():
    """get net"""
    reset(seed)
    net = LayerCase()
    return net


# 动态图结果
reset(seed)
net = _net_instant()
net.eval()
logit = net(*tensor_data)

# export
reset(seed)
st_net = paddle.jit.to_static(_net_instant())
st_net.eval()
st_net(*tensor_data)
paddle.jit.save(st_net, path=os.path.join("save_path", "jit_save"))

# infer
reset(seed)
config = paddle_infer.Config("save_path/jit_save" + ".pdmodel", "save_path/jit_save" + ".pdiparams")
config.enable_use_gpu(1000, 0)
predictor = paddle_infer.create_predictor(config)
input_names = predictor.get_input_names()
for i, name in enumerate(input_names):
    input_handle = predictor.get_input_handle(name)
    input_tmp = data[i]
    input_handle.copy_from_cpu(input_tmp)
predictor.run()
output_names = predictor.get_output_names()
if len(output_names) > 1:
    infer_res = []
    for i, name in enumerate(output_names):
        output_handle = predictor.get_output_handle(output_names[i])
        infer_res.append(output_handle.copy_to_cpu())
else:
    output_handle = predictor.get_output_handle(output_names[0])
    infer_res = output_handle.copy_to_cpu()

print(logit - infer_res)
