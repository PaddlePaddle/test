"""
Module Docstring:
This module contains functions for saving and loading models using PaddlePaddle.
"""
import os
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer
import numpy as np

# 生成随机数据
paddle.seed(33)
np.random.seed(32)


def save_model(model, path, inputs):
    """保存模型"""
    # 保存模型
    model.eval()
    save_model_path = os.path.join(path.split("/")[0], path.split("/")[1], "model.pdparams")
    paddle.save(model.state_dict(), save_model_path)
    # 加载模型的参数
    model.set_state_dict(paddle.load(save_model_path))
    pred = model(inputs)
    output3 = paddle.mean(pred)
    print("Paddle.Save Model Output Mean: " + str(output3.numpy()))
    return pred, output3


def jit_save(model, path, inputs):
    """paddle.jit.save"""
    # 设置训练模式
    model.eval()
    pred = model(inputs)
    outputs = paddle.mean(pred)
    print(f"Jit.Save Model Output Mean: {outputs.numpy()}")
    # 保存模型
    paddle.jit.save(model, path)

    return pred, outputs


def jit_load(model, path, inputs):
    """paddle.jit.load"""
    # 加载模型
    loaded_model = paddle.jit.load(path)

    pred = loaded_model(inputs)
    output_mean = paddle.mean(pred)
    print(f"Loaded Model Output Mean: {output_mean.numpy()}")

    return pred, output_mean


def infer(num_image, path, inputs):
    """推理"""
    # 创建 config
    model_name_or_path = path.split("/demo")[0]
    model_prefix = path.split("/")[-1]
    params_path = os.path.join(model_name_or_path, model_prefix + ".pdiparams")
    if paddle.framework.use_pir_api():
        model_file = os.path.join(model_name_or_path, model_prefix + ".json")
    else:
        model_file = os.path.join(model_name_or_path, model_prefix + ".pdmodel")
    # inference_config = paddle.inference.Config(model_path, params_path)

    config = paddle_infer.Config(model_file, params_path)
    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    # 设置输入  这里的64是输入图片的尺寸 需要和前面保持一致的
    input_handle.reshape([num_image, 3, 64, 64])
    input_handle.copy_from_cpu(inputs)
    # 运行 predictor
    predictor.run()
    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    pred = output_handle.copy_to_cpu()  # numpy.ndarray 类型
    outputs = np.mean(pred)
    print(f"Infer Output Mean: {outputs}")

    return pred, outputs


def are_close(a, b, tol=1e-3):
    """
    判断两个浮点数是否足够接近。

    a, b: 要比较的两个浮点数或numpy数组。
    tol: 容差，默认为1e-9。

    如果a和b足够接近，则返回True；否则返回False。
    """
    return np.allclose(a, b, atol=tol)


class RandomNet(nn.Layer):
    """一个简单的神经网络"""

    def __init__(self):
        super(RandomNet, self).__init__()
        # 定义卷积层和ReLU层
        self.layers = nn.LayerList(
            [
                nn.Conv2D(3, 116, kernel_size=[3, 3], padding=1),
                nn.ReLU(),
                nn.Conv2D(116, 124, kernel_size=[3, 3], padding=1),
                nn.BatchNorm2D(num_features=124, momentum=0.9, epsilon=1e-05),
                nn.ReLU(),
                nn.Conv2D(124, 99, kernel_size=[3, 3], padding=1),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
                nn.Conv2D(99, 117, kernel_size=[3, 3], padding=1),
                nn.ReLU(),
                nn.Conv2D(117, 110, kernel_size=[3, 3], padding=1),
                nn.ReLU(),
            ]
        )

        self.fc_layers = nn.LayerList([nn.Linear(in_features=112640, out_features=10)])

    @paddle.jit.to_static
    def forward(self, x):
        """前向传播函数"""
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2D):
                # 处理池化层输出，这里不需要额外操作
                pass
        x = paddle.reshape(paddle.flatten(x), [x.shape[0], -1])

        for layer1 in self.fc_layers:
            x = layer1(x)
        return x


# if __name__ == "__main__":

def test_model7():
    """test model"""

    # 转换为Paddle张量
    num_image = 10
    size = 64
    max_conv_layers = 5

    assert (size % (2**max_conv_layers)) == 0, "不能整除，有报错风险！"
    # assert (size % (2 ** max_conv_layers)) != 0, "不应该能够整除，但却整除了！"

    data = np.random.randn(num_image, 3, size, size).astype("float32")

    # label = np.random.randint(0, 10, (10, 1), dtype="int64")
    inputs = paddle.to_tensor(data)
    # labels = paddle.to_tensor(label)

    path = "simple/model7/demo7"  # 路径这里用demo，若改infer中对应需要改

    model = RandomNet()
    # 这里可以添加训练、保存、加载和预测的逻辑
    # print(model)

    # 模型保存 加载评估 预测部署
    pred_0, output_0 = save_model(model, path, inputs)
    pred_1, output_1 = jit_save(model, path, inputs)
    pred_2, output_2 = jit_load(model, path, inputs)
    pred_3, output_3 = infer(num_image, path, data)

    assert output_1 == output_0 and are_close(output_2, output_3), "模型运行失败，可能是数据不一致或计算错误"
    print("恭喜你,模型运行成功，再接再厉！！")
