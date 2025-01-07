# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
test_poiseuille_flow.py
"""
# 导入模块
import os
import sys
import subprocess

import pytest
import allure
import numpy as np

from tools.log_analysis import get_last_epoch_loss, get_last_eval_metric_for_darcy

@allure.title("功能测试")
def test_poiseuille_flow_exit_code():
    """
    测试函数：测试 darcy2d.py 脚本的退出码是否为 0 以保证可视化文件的正常保存
    """
    epoch_num = 10  # 迭代次数
    output_dir = "./output_poiseuille_flow"  # 输出目录
    py_version = os.getenv("py_version", "3.8")  # Python 版本号，从环境变量中获取，默认值为3.8

    # 执行命令行命令，运行 darcy2d.py 脚本
    command = f"python{py_version} ../../examples/pipe/poiseuille_flow.py \
               TRAIN.epochs={epoch_num} \
               output_dir={output_dir}"

    process = subprocess.Popen(command, shell=True)

    # 等待脚本执行完成，并返回退出码
    exit_code = process.wait()

    # 断言退出码为 0
    assert exit_code == 0

@allure.title("训练精度测试")
def test_poiseuille_flow_loss():
    """
    测试函数：测试 poiseuille_flow.py 脚本的损失值
    """
    epoch_num = 10  # 迭代次数
    output_dir = "./output_poiseuille_flow"  # 输出目录
    base_loss = 0.00036  # 基准损失值

    # 获取训练过程的日志文件并计算最后一轮迭代的损失值
    log_file = os.path.join(output_dir, "train.log")
    last_loss = get_last_epoch_loss(log_file, epoch_num)

    # 断言最后一轮迭代的损失值与基准
    assert np.allclose(float(last_loss), base_loss, rtol=1e-6)


if __name__ == "__main__":
    # 使用 pytest 模块运行测试函数
    code = pytest.main(["--alluredir=./allure", sys.argv[0]])
    sys.exit(code)
