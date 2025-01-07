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
from tools.tools import run_model, plot_kpi_curves, log_parse
from tools.get_result_html import get_html
from tools.send_email import send_email

test_data = {}
model_list = []
project_dir = os.getcwd()
with open('./model_daily.json', 'r', encoding='utf-8') as f:
    # 读取JSON数据
    test_json = json.load(f)
    # 遍历 JSON 对象获取所有的键
    for key in test_json.keys():
        model_list.append(key)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("pytorch")
def test_pytorch(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    # 执行torch
    pytorch_exit = run_model(model_name, 'pytorch')
    assert pytorch_exit == 0

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动态图")
def test_dynamic(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    # 初始化测试数据
    if model_name not in test_data:
        test_data[model_name]={}
    if "dynamic" not in test_data[model_name]:
        test_data[model_name]["dynamic"]={}
    if "L2" not in test_data[model_name]["dynamic"]:
        test_data[model_name]["dynamic"]["L2"] ={}
    if "L4" not in test_data[model_name]["dynamic"]:
        test_data[model_name]["dynamic"]["L4"] ={}
    test_data[model_name]["dynamic"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dynamic"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行动态图
    dynamic_exit = run_model(model_name, 'dynamic')
    if dynamic_exit != 0:
        test_data[model_name]["dynamic"]["L2"]["status"] = "Failed"
        test_data[model_name]["dynamic"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dynamic_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    kpi_loss['pytorch'] = pytorch_kpi
    # 解析动态图日志文件，提取损失值
    dynamic_kpi = log_parse(f'./logs/{model_name}/{model_name}_dynamic.log', 'Loss')
    kpi_loss['dynamic'] = dynamic_kpi
    if pytorch_kpi[0] == -1 or dynamic_kpi[0] == -1:
        test_data[model_name]["dynamic"]["L2"]["status"] = "Failed"
        test_data[model_name]["dynamic"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert pytorch_kpi[0] != -1
    assert dynamic_kpi[0] != -1
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dynamic')
    # 计算动态图和pytorch的损失值差异
    # diff_dynamic_1 = (dynamic_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    # diff_dynamic_all = sum((dynamic_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    # print("diff_dynamic_1:", diff_dynamic_1)
    # print("diff_dynamic_all:", diff_dynamic_all)
    # # 判断差异是否小于阈值
    # assert diff_dynamic_1 < 1e-6
    # assert diff_dynamic_all < 1e-5
    dynamic_kpi_avg = np.mean(dynamic_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dynamic"]["L2"]["rtol"] = np.abs(dynamic_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dynamic"]["L2"]["atol"] = np.abs(dynamic_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dynamic"]["L4"]["rtol"] = np.abs(dynamic_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dynamic"]["L4"]["atol"] = np.abs(dynamic_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dynamic"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dynamic"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dynamic"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dynamic"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dynamic"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dynamic"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dynamic"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dynamic"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dynamic_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dynamic_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动转静")
def test_dy2st(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    if model_name not in test_data:
        test_data[model_name]={}
    if "dy2st" not in test_data[model_name]:
        test_data[model_name]["dy2st"]={}
    if "L2" not in test_data[model_name]["dy2st"]:
        test_data[model_name]["dy2st"]["L2"] ={}
    if "L4" not in test_data[model_name]["dy2st"]:
        test_data[model_name]["dy2st"]["L4"] ={}
    test_data[model_name]["dy2st"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dy2st"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行dy2st
    dy2st_exit = run_model(model_name, 'dy2st')
    if dy2st_exit != 0:
        test_data[model_name]["dy2st"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    kpi_loss['pytorch'] = pytorch_kpi
    # 解析dy2st日志文件，提取损失值
    dy2st_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st.log', 'Loss')
    kpi_loss['dy2st'] = dy2st_kpi
    if dy2st_kpi[0] == -1 or pytorch_kpi[0] == -1:
        test_data[model_name]["dy2st"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_kpi[0] != -1
    assert pytorch_kpi[0] != -1
    kpi_loss['dy2st'] = dy2st_kpi
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st')
    # # 计算dy2st和pytorch的损失值差异
    # diff_dy2st_1 = (dy2st_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    # diff_dy2st_all = sum((dy2st_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    # print("diff_dy2st_1:", diff_dy2st_1)
    # print("diff_dy2st_all:", diff_dy2st_all)
    # assert diff_dy2st_1 < 1e-6
    # assert diff_dy2st_all < 1e-5
    dy2st_kpi_avg = np.mean(dy2st_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dy2st"]["L2"]["rtol"] = np.abs(dy2st_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dy2st"]["L2"]["atol"] = np.abs(dy2st_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dy2st"]["L4"]["rtol"] = np.abs(dy2st_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dy2st"]["L4"]["atol"] = np.abs(dy2st_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dy2st"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dy2st"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dy2st"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dy2st"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dy2st_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dy2st_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动转静+组合算子")
def test_dy2st_prim(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    if model_name not in test_data:
        test_data[model_name]={}
    if "dy2st_prim"not in test_data[model_name]:
        test_data[model_name]["dy2st_prim"]={}
    if "L2" not in test_data[model_name]["dy2st_prim"]:
        test_data[model_name]["dy2st_prim"]["L2"] ={}
    if "L4" not in test_data[model_name]["dy2st_prim"]:
        test_data[model_name]["dy2st_prim"]["L4"] ={}
    test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行dy2st+prim
    dy2st_prim_exit = run_model(model_name, 'dy2st_prim')
    if dy2st_prim_exit != 0:
        test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_prim_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    kpi_loss['pytorch'] = pytorch_kpi
    dy2st_prim_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim.log', 'Loss')
    kpi_loss['dy2st_prim'] = dy2st_prim_kpi
    if dy2st_prim_kpi[0] == -1 or pytorch_kpi[0] == -1:
        test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert pytorch_kpi[0] != -1
    assert dy2st_prim_kpi[0] != -1
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim')
    # # 计算动转静+prim和pytorch的损失值差异
    # diff_prim_1 = (dy2st_prim_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    # diff_prim_all = sum((dy2st_prim_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    # print("diff_prim_1:", diff_prim_1)
    # print("diff_prim_all:", diff_prim_all)
    # assert diff_prim_1 < 1e-6
    # assert diff_prim_all < 1e-5
    dy2st_prim_kpi_avg = np.mean(dy2st_prim_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dy2st_prim"]["L2"]["rtol"] = np.abs(dy2st_prim_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dy2st_prim"]["L2"]["atol"] = np.abs(dy2st_prim_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dy2st_prim"]["L4"]["rtol"] = np.abs(dy2st_prim_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dy2st_prim"]["L4"]["atol"] = np.abs(dy2st_prim_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dy2st_prim"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dy2st_prim"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dy2st_prim_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dy2st_prim_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动转静+组合算子+CSE")
def test_dy2st_prim_cse(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    if model_name not in test_data:
        test_data[model_name]={}
    if "dy2st_prim_cse"not in test_data[model_name]:
        test_data[model_name]["dy2st_prim_cse"]={}
    if "L2" not in test_data[model_name]["dy2st_prim_cse"]:
        test_data[model_name]["dy2st_prim_cse"]["L2"] ={}
    if "L4" not in test_data[model_name]["dy2st_prim_cse"]:
        test_data[model_name]["dy2st_prim_cse"]["L4"] ={}
    test_data[model_name]["dy2st_prim_cse"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dy2st_prim_cse"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行dy2st+prim+cse
    dy2st_prim_cse_exit = run_model(model_name, 'dy2st_prim_cse')
    if dy2st_prim_cse_exit != 0:
        test_data[model_name]["dy2st_prim"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_prim_cse_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    kpi_loss['pytorch'] = pytorch_kpi
    dy2st_prim_cse_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim_cse.log', 'Loss')
    kpi_loss['dy2st_prim_cse'] = dy2st_prim_cse_kpi
    if dy2st_prim_cse_kpi[0] == -1 or pytorch_kpi[0] == -1:
        test_data[model_name]["dy2st_prim_cse"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim_cse"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_prim_cse_kpi[0] != -1
    assert pytorch_kpi[0] != -1
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim_cse')
    # # 计算动转静+prim和pytorch的损失值差异
    # diff_prim_1 = (dy2st_prim_cse_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    # diff_prim_all = sum((dy2st_prim_cse_kpi - pytorch_kpi)/pytorch_kpi)/len(pytorch_kpi)
    # print("diff_prim_1:", diff_prim_1)
    # print("diff_prim_all:", diff_prim_all)
    # assert diff_prim_1 < 1e-6
    # assert diff_prim_all < 1e-5
    dy2st_prim_cse_kpi_avg = np.mean(dy2st_prim_cse_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dy2st_prim_cse"]["L2"]["rtol"] = np.abs(dy2st_prim_cse_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dy2st_prim_cse"]["L2"]["atol"] = np.abs(dy2st_prim_cse_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dy2st_prim_cse"]["L4"]["rtol"] = np.abs(dy2st_prim_cse_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dy2st_prim_cse"]["L4"]["atol"] = np.abs(dy2st_prim_cse_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dy2st_prim_cse"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cse"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cse"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cse"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dy2st_prim_cse"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cse"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cse"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cse"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dy2st_prim_cse_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dy2st_prim_cse_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动转静+组合算子+CINN")
def test_dy2st_prim_cinn(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    if model_name not in test_data:
        test_data[model_name]={}
    if "dy2st_prim_cinn"not in test_data[model_name]:
        test_data[model_name]["dy2st_prim_cinn"]={}
    if "L2" not in test_data[model_name]["dy2st_prim_cinn"]:
        test_data[model_name]["dy2st_prim_cinn"]["L2"] ={}
    if "L4" not in test_data[model_name]["dy2st_prim_cinn"]:
        test_data[model_name]["dy2st_prim_cinn"]["L4"] ={}
    test_data[model_name]["dy2st_prim_cinn"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dy2st_prim_cinn"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行dy2st+prim+cinn
    dy2st_prim_cinn_exit = run_model(model_name, 'dy2st_prim_cinn')
    if dy2st_prim_cinn_exit != 0:
        test_data[model_name]["dy2st_prim_cinn"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim_cinn"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_prim_cinn_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')

    kpi_loss['pytorch'] = pytorch_kpi
    dy2st_prim_cinn_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim_cinn.log', 'Loss')
    kpi_loss['dy2st_prim_cinn'] = dy2st_prim_cinn_kpi
    if dy2st_prim_cinn_kpi[0] == -1 or pytorch_kpi[0] == -1:
        test_data[model_name]["dy2st_prim_cinn"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim_cinn"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert pytorch_kpi[0] != -1
    assert dy2st_prim_cinn_kpi[0] != -1
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim_cinn')
    dy2st_prim_cinn_kpi_avg = np.mean(dy2st_prim_cinn_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dy2st_prim_cinn"]["L2"]["rtol"] = np.abs(dy2st_prim_cinn_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dy2st_prim_cinn"]["L2"]["atol"] = np.abs(dy2st_prim_cinn_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dy2st_prim_cinn"]["L4"]["rtol"] = np.abs(dy2st_prim_cinn_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dy2st_prim_cinn"]["L4"]["atol"] = np.abs(dy2st_prim_cinn_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dy2st_prim_cinn"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cinn"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cinn"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cinn"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dy2st_prim_cinn"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cinn"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cinn"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cinn"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dy2st_prim_cinn_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dy2st_prim_cinn_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

@pytest.mark.parametrize("model_name", model_list)
@allure.title("动转静+组合算子+CINN+CSE")
def test_dy2st_prim_cinn_cse(model_name):
    # 切换到项目根目录
    os.chdir(project_dir)
    if model_name not in test_data:
        test_data[model_name]={}
    if "dy2st_prim_cinn_cse"not in test_data[model_name]:
        test_data[model_name]["dy2st_prim_cinn_cse"]={}
    if "L2" not in test_data[model_name]["dy2st_prim_cinn_cse"]:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L2"] ={}
    if "L4" not in test_data[model_name]["dy2st_prim_cinn_cse"]:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L4"] ={}
    test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["status"] = "Timeout"
    test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["status"] = "Timeout"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    # 执行dy2st+prim+cinn
    dy2st_prim_cinn_cse_exit = run_model(model_name, 'dy2st_prim_cinn_cse')
    if dy2st_prim_cinn_cse_exit != 0:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert dy2st_prim_cinn_cse_exit == 0
    kpi_loss = {}
    # 解析pytorch的日志文件，提取损失值
    pytorch_kpi = log_parse(f'./logs/{model_name}/{model_name}_pytorch.log', 'Loss')
    kpi_loss['pytorch'] = pytorch_kpi
    dy2st_prim_cinn_cse_kpi = log_parse(f'./logs/{model_name}/{model_name}_dy2st_prim_cinn_cse.log', 'Loss')
    kpi_loss['dy2st_prim_cinn_cse'] = dy2st_prim_cinn_cse_kpi
    if dy2st_prim_cinn_cse_kpi[0] == -1 or pytorch_kpi[0] == -1:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["status"] = "Failed"
        test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    assert pytorch_kpi[0] != -1
    assert dy2st_prim_cinn_cse_kpi[0] != -1
    # 绘制KPI曲线图
    plot_kpi_curves(model_name, kpi_loss, 'dy2st_prim_cinn_cse')
    dy2st_prim_cinn_cse_kpi_avg = np.mean(dy2st_prim_cinn_cse_kpi)
    pytorch_kpi_avg = np.mean(pytorch_kpi)
    test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["rtol"] = np.abs(dy2st_prim_cinn_cse_kpi[0] - pytorch_kpi[0])/pytorch_kpi[0]
    test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["atol"] = np.abs(dy2st_prim_cinn_cse_kpi[0] - pytorch_kpi[0])
    test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["rtol"] = np.abs(dy2st_prim_cinn_cse_kpi_avg - pytorch_kpi_avg)/pytorch_kpi_avg
    test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["atol"] = np.abs(dy2st_prim_cinn_cse_kpi_avg - pytorch_kpi_avg)
    if test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L2"]["status"] = "Failed"
    if test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["atol"] < 1e-5 and test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["rtol"] < 1.3e-6:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["status"] = "Success"
    else:
        test_data[model_name]["dy2st_prim_cinn_cse"]["L4"]["status"] = "Failed"
    with open('./test_data.json', 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)
    np.testing.assert_allclose(dy2st_prim_cinn_cse_kpi[0], pytorch_kpi[0], atol=1e-5, rtol=1.3e-6)
    np.testing.assert_allclose(dy2st_prim_cinn_cse_kpi_avg, pytorch_kpi_avg, atol=1e-5, rtol=1.3e-6)

if __name__ == "__main__":
    current_date = datetime.now()
    code = pytest.main(["--json=test.json", "--alluredir=./allure", "--html=report.html", "--timeout=1800", sys.argv[0]])
    if not os.path.exists("html_result"):
        os.makedirs("html_result") 
    get_html("./test_data.json","./html_result/index.html")
    send_email('auto_send@baidu.com',['suijiaxin@baidu.com', 'hesensen@baidu.com', 'chenxiaoxu02@baidu.com', 'chenxi67@baidu.com', 'chenyaowen02@baidu.com', 'chenzhuo13@baidu.com', 'liuhongyu02@baidu.com', 'tianchao04@baidu.com', 'wanghao107@baidu.com', 'xiongkun03@baidu.com', 'xuxiaojian@baidu.com', 'zhangliujie@baidu.com', 'zhangyanbo02@baidu.com', 'huxiaoguang@baidu.com', 'zhengtianyu@baidu.com'],'Modulus 天级别例行测试报告')
    sys.exit(code)
