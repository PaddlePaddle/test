"""
Module Docstring:
This module contains functions for saving and loading models using PaddlePaddle.
"""
import os
import gc
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer
import numpy as np


# 生成随机数据
paddle.seed(33)
np.random.seed(32)


def create_random_model(num_classes=10, max_conv_layers=3, max_fc_layers=2):
    """创建模型"""

    class RandomNet(nn.Layer):
        """随机网络"""

        def __init__(self, input_height, input_width):
            super(RandomNet, self).__init__()
            self.layers = nn.LayerList()
            self.fc_layers = nn.LayerList()
            self.input_height = input_height
            self.input_width = input_width

            # 随机添加卷积层和ReLU
            num_conv_layers = np.random.randint(1, max_conv_layers + 1)
            in_channels = 3
            pool_count = 0
            for _ in range(num_conv_layers):
                out_channels = np.random.randint(64, 128)
                self.layers.append(
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        data_format="NCHW",
                    )
                )
                if np.random.rand() > 0.6:  # 50% 概率添加BatchNorm2D
                    self.layers.append(nn.BatchNorm2D(out_channels))

                self.layers.append(nn.ReLU())
                in_channels = out_channels
                if np.random.rand() > 0.5:  # 50% 概率添加池化层
                    self.layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
                    pool_count += 1

            # 检查是否需要添加一个池化层
            if pool_count == 0:
                self.layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
                pool_count = 1  # 或者不需要显式设置，因为后续不再使用pool_count计算flatten_size

            # 计算全连接层前的特征图大小
            self.flatten_size = in_channels * (input_height // (2**pool_count)) * (input_width // (2**pool_count))
            # print(f"in_channels:{in_channels}")
            # print(input_height // (2 ** pool_count))
            # print(input_width // (2 ** pool_count))
            # print(f"pool_count:{pool_count}")
            assert self.flatten_size > 0, "Flatten size is zero, check input dimensions and pooling count"

            # 添加全连接层
            num_fc_layers = np.random.randint(1, max_fc_layers + 1)
            fc_sizes = [np.random.randint(128, 1024) for _ in range(num_fc_layers - 1)]
            fc_sizes.append(num_classes)

            for i in range(num_fc_layers):
                if i == 0:
                    in_features = self.flatten_size
                else:
                    in_features = fc_sizes[i - 1]
                self.fc_layers.append(nn.Linear(in_features=in_features, out_features=fc_sizes[i]))
                if i < num_fc_layers - 1:
                    self.fc_layers.append(nn.ReLU())

        @paddle.jit.to_static
        def forward(self, x):
            """前向传播"""
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2D):
                    # 处理池化层输出，这里不需要额外操作
                    pass
            x = paddle.reshape(paddle.flatten(x), [x.shape[0], -1])

            for layer1 in self.fc_layers:
                x = layer1(x)
            return x

    return RandomNet(64, 64)


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


def remove_data_format(layer_str):
    """移除某部分"""
    parts = layer_str.split(",")
    # 过滤掉包含 'data_format=NCHW' 的部分
    filtered_parts = [part.strip() for part in parts if "data_format=NCHW" not in part]
    # 重新连接剩余的部分
    filtered_parts = ", ".join(filtered_parts)
    if not filtered_parts.endswith(")"):
        filtered_parts += ")"

    return filtered_parts


def remove_data_float(layer_str):
    """移除某部分"""
    parts = layer_str.split(",")
    # 过滤掉包含 'data_format=NCHW' 的部分
    filtered_parts = [part.strip() for part in parts if "float32" not in part]
    # 重新连接剩余的部分
    filtered_parts = ", ".join(filtered_parts)
    if not filtered_parts.endswith(")"):
        filtered_parts += ")"

    return filtered_parts


def write_model(layers, fc_layers, i, save_model=True):
    """写入模块"""
    # 将列表中的每个元素转换为字符串，并用换行符连接
    a_str = ",\n            ".join(f"nn.{remove_data_format(a)}" for a in map(str, layers))
    b_str = ",\n            ".join(f"nn.{remove_data_float(b)}" for b in map(str, fc_layers))

    # 模板字符串，使用 {} 作为占位符
    template = """

# Module Docstring:
# This module contains functions for saving and loading models using PaddlePaddle.

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

def save_model(model, path,inputs):
    # 保存模型
    model.eval()

    save_model_path = os.path.join(path.split("/")[0],path.split("/")[1],"model.pdparams")

    paddle.save(model.state_dict(),save_model_path)

    #加载模型的参数
    model.set_state_dict(paddle.load(save_model_path))
    pred = model(inputs)
    output3 = paddle.mean(pred)
    print("Paddle.Save Model Output Mean: "+str(output3.numpy()))
    return pred,output3

def jit_save(model,path,inputs):
    # 设置训练模式
    model.eval()
    pred = model(inputs)
    outputs_mean = paddle.mean(pred)
    print("Save Model Output Mean: "+ str(outputs_mean.numpy()))
    # 保存模型

    paddle.jit.save(model, path)


    return pred,outputs_mean

def jit_load(model,path,inputs):
    # 加载模型
    loaded_model = paddle.jit.load(path)

    pred = loaded_model(inputs)
    output_mean = paddle.mean(pred)
    print("Loaded Model Output Mean: "+str(output_mean.numpy()))

    return pred,output_mean

def infer(num_image,path,inputs):
    # 创建 config
    model_name_or_path = path.split("/demo")[0]
    model_prefix = path.split("/")[-1]
    params_path = os.path.join(model_name_or_path,model_prefix + ".pdiparams")
    if paddle.framework.use_pir_api():
        model_file = os.path.join(model_name_or_path,model_prefix + ".json")
    else:
        model_file = os.path.join(model_name_or_path,model_prefix + ".pdmodel")
    # inference_config = paddle.inference.Config(model_path, params_path)

    config = paddle_infer.Config(model_file, params_path)
    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    # 设置输入
    input_handle.reshape([num_image, 3, 64, 64])
    input_handle.copy_from_cpu(inputs)
    # 运行 predictor
    predictor.run()
    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    pred = output_handle.copy_to_cpu() # numpy.ndarray 类型
    outputs = np.mean(pred)
    print("Infer Output Mean: "+str(np.mean(pred)))

    return pred,outputs

def are_close(a, b, tol=1e-3):

    # 判断两个浮点数a和b是否足够接近，是则返回True；否则返回False；
    # a, b: 要比较的两个浮点数或numpy数组；
    # tol: 容差，默认为1e-9。
    return np.allclose(a, b, atol=tol)

class RandomNet(nn.Layer):
    def __init__(self):
        super(RandomNet, self).__init__()
        # 定义卷积层和ReLU层
        self.layers = nn.LayerList([
            {a_str}
        ])

        self.fc_layers = nn.LayerList([
            {b_str}
        ])

    @paddle.jit.to_static
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2D):
                # 处理池化层输出，这里不需要额外操作
                pass
        x = paddle.reshape(paddle.flatten(x), [x.shape[0], -1])

        for layer1 in self.fc_layers:
            x = layer1(x)
        return x

if __name__ == "__main__":

    # 转换为Paddle张量
    num_image = 10
    size = 64
    max_conv_layers = 5

    assert (size % (2 ** max_conv_layers)) == 0, "不能整除，有报错风险！"
    # assert (size % (2 ** max_conv_layers)) != 0, "不应该能够整除，但却整除了！"

    data = np.random.randn(num_image, 3, size, size).astype('float32')
    label = np.random.randint(0, 10, (10, 1), dtype='int64')
    inputs = paddle.to_tensor(data)
    labels = paddle.to_tensor(label)
    path = f"simple/model{c}/demo{d}" # 路径这里用demo，若改infer中对应需要改

    model = RandomNet()
    # 这里可以添加训练、保存、加载和预测的逻辑
    # print(model)

    # 模型保存 加载评估 预测部署
    pred_0,output_0 = save_model(model,path,inputs)
    pred_1,output_1 = jit_save(model,path,inputs)
    pred_2,output_2 = jit_load(model,path,inputs)
    pred_3,output_3 = infer(num_image,path,data)

    assert are_close(output_1, output_0) and are_close(output_2, output_3), "Failed, possibly due to inconsistent data or computational errors."
    print(f"Congratulations! The model run was successful. Keep up the good work!")

"""
    path = f"model{i+1}.py"
    if save_model is True:
        with open(path, "w+") as f:
            f.write(template.format(a_str=a_str, b_str=b_str, c=i + 1, d=i + 1))


# if __name__ == "__main__":

def test_model():
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

    # path = "simple/demo" #

    for i in range(num_models):
        model = create_random_model(num_classes=10, max_conv_layers=max_conv_layers, max_fc_layers=5)
        # 这里可以添加训练、保存、加载和预测的逻辑
        print(f"Model {i+1} Structure:")
        # print(model)

        path = f"simple/model{i+1}/demo{i+1}"  # 路径这里用demo，若改infer中对应需要改
        write_model(model.layers, model.fc_layers, i, save_model=True)

        # 模型保存 加载评估 预测部署
        pred_0, output_0 = save_model(model, path, inputs)
        pred_1, output_1 = jit_save(model, path, inputs)
        pred_2, output_2 = jit_load(model, path, inputs)
        pred_3, output_3 = infer(num_image, path, data)

        assert are_close(output_1, output_0) and are_close(
            output_2, output_3
        ), "Failed, possibly due to inconsistent data or computational errors."
        print(f"Congratulations! The {(i+1)}th model run was successful. Keep up the good work!")

        # 清理模型
        del model  # 目的是为了释放显存，目前尚未验证是否有效
        gc.collect()  # 尝试回收垃圾
