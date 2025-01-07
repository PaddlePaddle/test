import os
import re
import json
import sys
import time
import subprocess
import pytest
import allure
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_model(model_name, run_mode, extra_parameters='training.max_steps=100'):
    """
    运行模型，并返回模型的输出结果
    Args:
        model_name (str): 模型名称，如resnet50
        run_mode (str): 运行模式，如dynamic、dy2st、dy2st_prim
        extra_parameters (str): 额外参数，如training.max_steps=100 默认为steps=100
    """
    allure.dynamic.sub_suite(model_name.replace("^", "/"))
    with open('./model.json', 'r', encoding='utf-8') as f:
    # 读取JSON数据
        model_json = json.load(f)
    model_location = model_json[model_name]
    model_file = model_location.split("/")[-1]
    model_path = model_location.rsplit("/", 1)[0]
    os.system("export CUDA_VISIBLE_DEVICES=0")
    os.system("fuser -k /dev/nvidia*")
    if extra_parameters == 'training.max_steps=100':
        os.environ['debug'] = '1'
    if run_mode == 'dynamic':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))  
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"]) 
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"])
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)
        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['to_static'] = 'False'
        os.environ['FLAGS_use_cinn'] = 'False'
        os.environ['FLAGS_prim_all'] = 'False'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        os.environ ['FLAGS_enable_cse_in_dy2st']= '0'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    to_static=False \
                                    FLAGS_use_cinn=False \
                                    FLAGS_prim_all=False \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=0 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode == 'dy2st':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"])
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"])
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)
        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['to_static'] = 'True'
        os.environ['FLAGS_prim_all'] = 'False'
        os.environ['FLAGS_use_cinn'] = 'False'
        os.environ['FLAGS_enable_pir_in_executor'] = 'true'
        os.environ['FLAGS_enable_pir_api'] = 'True'
        os.environ['FLAGS_cinn_bucket_compile'] = 'True'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_nvrtc_compile_to_cubin'] = 'True'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        os.environ ['FLAGS_enable_cse_in_dy2st']= '0'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    to_static=True \
                                    FLAGS_prim_all=False \
                                    FLAGS_use_cinn=False \
                                    FLAGS_enable_pir_in_executor=true \
                                    FLAGS_enable_pir_api=True \
                                    FLAGS_cinn_bucket_compile=True \
                                    FLAGS_group_schedule_tiling_first=1 \
                                    FLAGS_cinn_new_group_scheduler=1 \
                                    FLAGS_nvrtc_compile_to_cubin=True \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=0 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode == 'dy2st_prim':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"])
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"]) 
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)
        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['FLAGS_prim_vjp_skip_default_ops'] = 'False'
        os.environ['to_static'] = 'True'
        os.environ['FLAGS_prim_all'] = 'True'
        os.environ['FLAGS_use_cinn'] = 'False'
        os.environ['FLAGS_enable_pir_in_executor'] = 'true'
        os.environ['FLAGS_enable_pir_api'] = 'True'
        os.environ['FLAGS_cinn_bucket_compile'] = 'True'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_nvrtc_compile_to_cubin'] = 'True'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        os.environ ['FLAGS_enable_cse_in_dy2st']= '0'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    FLAGS_prim_vjp_skip_default_ops=False \
                                    to_static=True \
                                    FLAGS_prim_all=True \
                                    FLAGS_use_cinn=False \
                                    FLAGS_enable_pir_in_executor=true \
                                    FLAGS_enable_pir_api=True \
                                    FLAGS_cinn_bucket_compile=True \
                                    FLAGS_group_schedule_tiling_first=1 \
                                    FLAGS_cinn_new_group_scheduler=1 \
                                    FLAGS_nvrtc_compile_to_cubin=True \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=0 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode == 'dy2st_prim_cse':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"])
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"]) 
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)
        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['FLAGS_prim_vjp_skip_default_ops'] = 'False'
        os.environ['to_static'] = 'True'
        os.environ['FLAGS_prim_all'] = 'True'
        os.environ['FLAGS_use_cinn'] = 'False'
        os.environ['FLAGS_enable_pir_in_executor'] = 'true'
        os.environ['FLAGS_enable_pir_api'] = 'True'
        os.environ['FLAGS_cinn_bucket_compile'] = 'True'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_nvrtc_compile_to_cubin'] = 'True'
        os.environ['FLAGS_enable_cse_in_dy2st'] = '1'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    FLAGS_prim_vjp_skip_default_ops=False \
                                    to_static=True \
                                    FLAGS_prim_all=True \
                                    FLAGS_use_cinn=False \
                                    FLAGS_enable_pir_in_executor=true \
                                    FLAGS_enable_pir_api=True \
                                    FLAGS_cinn_bucket_compile=True \
                                    FLAGS_group_schedule_tiling_first=1 \
                                    FLAGS_cinn_new_group_scheduler=1 \
                                    FLAGS_nvrtc_compile_to_cubin=True \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=1 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode == 'dy2st_prim_cinn':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"])
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"])
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)

        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['FLAGS_prim_vjp_skip_default_ops'] = 'False'
        os.environ['to_static'] = 'True'
        os.environ['FLAGS_use_cinn'] = 'True'
        os.environ['FLAGS_prim_all'] = 'True'
        os.environ['FLAGS_enable_pir_in_executor'] = 'true'
        os.environ['FLAGS_enable_pir_api'] = 'True'
        os.environ['FLAGS_cinn_bucket_compile'] = 'True'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_nvrtc_compile_to_cubin'] = 'True'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        os.environ ['FLAGS_enable_cse_in_dy2st']= '0'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    FLAGS_prim_vjp_skip_default_ops=False \
                                    to_static=True \
                                    FLAGS_prim_all=True \
                                    FLAGS_use_cinn=True \
                                    FLAGS_enable_pir_in_executor=true \
                                    FLAGS_enable_pir_api=True \
                                    FLAGS_cinn_bucket_compile=True \
                                    FLAGS_group_schedule_tiling_first=1 \
                                    FLAGS_cinn_new_group_scheduler=1 \
                                    FLAGS_nvrtc_compile_to_cubin=True \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=0 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode == 'dy2st_prim_cinn_cse':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # subprocess.run(["git", "checkout", "modified_paddle_dy2st"])
        # print("\n modified_paddle_dy2st commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"])
        checkout_branch('modified_paddle_dy2st', model_name)
        download_datasets()
        os.chdir(current_dir)

        os.environ['loss_monitor'] = '1'
        os.environ['loss_monitor_pytorch_paddle'] = '1'
        os.environ['FLAGS_prim_vjp_skip_default_ops'] = 'False'
        os.environ['to_static'] = 'True'
        os.environ['FLAGS_use_cinn'] = 'True'
        os.environ['FLAGS_prim_all'] = 'True'
        os.environ['FLAGS_enable_pir_in_executor'] = 'true'
        os.environ['FLAGS_enable_pir_api'] = 'True'
        os.environ['FLAGS_cinn_bucket_compile'] = 'True'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_nvrtc_compile_to_cubin'] = 'True'
        os.environ['FLAGS_enable_cse_in_dy2st'] = '1'
        os.environ['FLAGS_enable_auto_recompute'] = '1'
        os.environ['load_data'] = 'True'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    loss_monitor=1 \
                                    loss_monitor_pytorch_paddle=1 \
                                    FLAGS_prim_vjp_skip_default_ops=False \
                                    to_static=True \
                                    FLAGS_prim_all=True \
                                    FLAGS_use_cinn=True \
                                    FLAGS_enable_pir_in_executor=true \
                                    FLAGS_enable_pir_api=True \
                                    FLAGS_cinn_bucket_compile=True \
                                    FLAGS_group_schedule_tiling_first=1 \
                                    FLAGS_cinn_new_group_scheduler=1 \
                                    FLAGS_nvrtc_compile_to_cubin=True \
                                    FLAGS_enable_auto_recompute=1 \
                                    load_data=True \
                                    FLAGS_enable_cse_in_dy2st=1 \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    elif run_mode =='pytorch':
        # 获取当前工作目录
        current_dir = os.getcwd()
        # 切换到上一级目录
        os.chdir(os.path.abspath(os.path.join(current_dir, os.pardir)))
        # # 执行 git checkout modified_torch 命令
        # subprocess.run(["git", "checkout", "modified_torch"])
        # os.system("rm -rf ./outputs")
        # print("\n modified_torch commit:")
        # subprocess.run(["git", "rev-parse", "HEAD"])
        # print("\n paddlepaddle commit:")
        # os.system('python -c "import paddle; paddle.version.show()"')
        checkout_branch('modified_torch', model_name)
        download_datasets()
        os.chdir(current_dir)
        os.environ['save_init_weight_data'] ='True'
        allure.dynamic.description(f"运行命令：debug=1 \
                                    save_init_weight_data=True \
                                    python {model_file} cuda_graphs=false jit=false jit_use_nvfuser=false graph.func_arch=false graph.func_arch_allow_partial_hessian=false {extra_parameters}")
    else:
        raise ValueError("模型运行方式错误：只能选择 动态图、动转静、动转静+prim、动转静+prim+cinn")
    # 创建日志目录
    current_dir = os.getcwd()
    log_dir = f'{current_dir}/logs/{model_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.chdir(f'../{model_path}')
    command = f"python {model_file} \
                cuda_graphs=false \
                jit=false \
                jit_use_nvfuser=false \
                graph.func_arch=false \
                graph.func_arch_allow_partial_hessian=false \
                {extra_parameters} >{log_dir}/{model_name}_{run_mode}.log 2>&1"
    process = subprocess.Popen(command, shell=True)
    exit_code = process.wait()
    diy_allure(f'{log_dir}/{model_name}_{run_mode}.log', f'{model_name}_{run_mode}', model_name)
    os.chdir(current_dir)
    return exit_code

def diy_allure(log_file_path, log_name, model_name="others"):
    # 读取日志文件内容
    log_content = ''
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            log_content = file.read()
    # 将日志文件内容附加到测试步骤中
    allure.attach(log_content, name=log_name, attachment_type=allure.attachment_type.TEXT)
    # allure.dynamic.sub_suite(model_name.replace("^", "/"))
    return 0
def checkout_branch(branch_name, model_name="lime"):
    """
    切换到指定分支
    """
    if os.path.exists('env_json.json'):
        with open('env_json.json', 'r', encoding='utf-8') as f:
            env_json = json.load(f)
    else:
        env_json = {}

    subprocess.run(["git", "checkout", "--", "examples/README.md"])
    subprocess.run(["git", "checkout", branch_name])
    os.system("rm -rf ./outputs")
    print(f"\n {branch_name} commit:")
    git_output = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(git_output)
    env_json[f'{branch_name}_commit'] = git_output
    if branch_name == 'modified_torch':
        paddle_version_output = os.popen('python -c "import paddle; paddle.version.show()"').read().strip()
        print("\n paddlepaddle commit:")
        print(paddle_version_output)
        env_json[f'PaddlePaddle'] = paddle_version_output
    with open("env_json.json", "w") as f:
        json.dump(env_json, f, indent=4)
    # elif branch_name == 'modified_paddle' and "lime" in model_name:
    #     if not os.path.exists("examples_sym.zip"):
    #         os.system("wget https://paddle-org.bj.bcebos.com/paddlescience/datasets/modulus/examples_sym.zip")
    #     if not os.path.exists("examples_sym"):
    #         os.system("unzip examples_sym.zip")
    #     os.system("unalias cp 2>/dev/null")
    #     os.system("cp -r -f -v ./examples_sym/examples/* ./examples/")

def download_datasets():
    """
    在torch运行之前部分模型需要下载必要的数据集
    """
    if not os.path.exists("./examples/darcy/datasets"):
        os.system("mkdir -p ./examples/darcy/datasets")
        os.system("wget https://paddle-qa.bj.bcebos.com/benchmark/pretrained/Darcy_241.tar.gz")
        os.system("tar -xf Darcy_241.tar.gz -C ./examples/darcy/datasets")

def plot_kpi_curves(model_name, model_data,run_mode):
    """
    绘制模型loss曲线对比图
    """
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))
    
    # 遍历JSON中的每个键值对
    for key, values in model_data.items():
        # 绘制曲线
        plt.plot(range(1, len(values) + 1), values, label=key)
    
    # 添加图例
    plt.legend()
    
    # 添加标题和标签
    plt.title(f'{model_name} {run_mode} Curves')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    # 保存图形到指定目录
    output_path = os.path.join(f"./logs/{model_name}/", f'{model_name}_{run_mode}_loss_curve.png')
    plt.savefig(output_path)
    # 清空图形
    plt.close()

def log_parse(log_content, kpi_name):
    """
    解析日志文件，提取指定KPI的值
    """
    # 定义一个空列表来存储提取到的KPI值
    kpi_value_all = []
    # 打开日志文件
    with open(log_content, 'r') as f:
        # 逐行读取日志内容
        for line in f.readlines():
            # 使用正则表达式匹配KPI名称和对应的数值，将提取到的数值添加到kpi_value_all中
            if kpi_name in line:
                regexp = r"%s\s*([0-9.]+)" % re.escape(kpi_name)  # 修改正则表达式以支持带有"."的KPI名称
                r = re.findall(regexp, line)
                if len(r) > 0:
                    kpi_value_all.append(float(r[0].strip()))
    
    # 如果没有提取到任何KPI值，则将最终的KPI值设置为-1
    if len(kpi_value_all) == 0:
        kpi_value_all.append(-1)
    # 否则，根据不同的KPI名称选择最后一个值作为最终的KPI值
    # 返回最终的KPI值
    kpi_value_all = np.array(kpi_value_all)
    return kpi_value_all
