#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
常用统计学计算策略
"""

import numpy as np
import matplotlib.pyplot as plt
from strategy.compare import base_compare


# 多种统计学计算策略
def trimmean(data_list, ratio=0.2):
    """
    掐头去尾求平均
    :param data_list: 输入的data list, 多次试验的结果集合
    """
    head = int(len(data_list) * ratio)
    tail = int(len(data_list) - len(data_list) * ratio)
    res = sum(sorted(data_list)[head:tail]) / (tail - head)
    return res


def mean(data_list):
    """
    求平均值
    :param data_list:
    :return:
    """
    res = sum(data_list) / len(data_list)
    return res


def best(data_list):
    """
    找出耗时最少的一次试验结果
    :param data_list: 输入的data list, 多次试验的结果集合
    :return: 最少的时间
    """
    # print('data_list is: ', data_list)
    res = min(data_list)
    return res


def best_top_k(data_list, ratio=0.2):
    """
    求最优top k的平均值，默认ratio=0.2
    :param data_list: 输入的data list, 多次试验的结果集合
    """
    head = int(len(data_list) * ratio)
    res = sum(sorted(data_list)[:head]) / head
    return res


# list等分
def split_list(lst, n):
    """
    将列表按顺序划分为 n 份
    Args:
        lst (list): 待划分的列表
        n (int): 划分的份数
    Returns:
        res (list): 划分后的列表，其中每个元素为原列表的 1/n 部分
    """
    if not isinstance(lst, list) or not isinstance(n, int) or len(lst) == 0 or n <= 0:
        return []
    quotient, remainder = divmod(len(lst), n)
    res = [[] for _ in range(n)]
    for i, value in enumerate(lst):
        index = i % n
        res[index].append(value)
    return res


# 作图
def perf_by_step(data_list, step_scale, filename):
    """
    横坐标为执行轮数, 纵坐标为耗时
    data_list: 耗时list, 以轮数为index
    step_scale: 以轮数的一定比例作图, list
    filename: 图片文件名
    """

    for percent_scale in step_scale:
        length = len(data_list)
        percent_count = int(length * percent_scale)
        # 提取横坐标（索引）和纵坐标（元素值）
        x_values = range(percent_count)
        y_values = data_list[:percent_count]

        # 绘制图形
        plt.clf()
        plt.plot(x_values, y_values, marker="o")  # 使用圆点作为数据点的标记
        plt.xlabel("Step-{}".format(percent_count))  # 设置横坐标轴标签
        plt.ylabel("Time-Consuming")  # 设置纵坐标轴标签
        plt.title(filename)  # 设置图形标题
        plt.grid(True)  # 显示网格线
        plt.savefig("{}_step{}.png".format(filename, percent_count))
        plt.clf()


# 计算四分位范围
def Q1_Q4_range(data_list):
    """
    取四分位数
    """
    # 按升序排列
    sorted_data = sorted(data_list)

    # 计算第一四分位数 Q1
    Q1 = np.percentile(sorted_data, 25)

    # 计算第三四分位数 Q3
    Q3 = np.percentile(sorted_data, 75)

    # 计算四分位数范围 IQR
    IQR = Q3 - Q1

    # 确定取值范围
    k = 1.5  # 可根据需要调整
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return lower_bound, upper_bound


def gsb_ratio_rule(res, single_gsb_dict={"G": 0, "S": 0, "B": 0, "error": 0}):
    """
    评分标准
    :param res: compare_res对比
    :return:
    """
    # res = base_compare.perf_compare(baseline, latest)

    try:
        res = res.rstrip("%")
        res = float(res) / 100
    except Exception:
        res = "error"

    if res == "error":
        single_gsb_dict["error"] += 1
    else:
        if res <= -0.05:
            single_gsb_dict["B"] += 1
        elif -0.05 < res <= 0.05:
            single_gsb_dict["S"] += 1
        elif res > 0.05:
            single_gsb_dict["G"] += 1
    return single_gsb_dict


def gsb_count_rule(res, single_gsb_dict={"G": 0, "S": 0, "B": 0, "error": 0}):
    """
    评分标准
    :param res: compare_res对比
    :return:
    """
    # res = base_compare.perf_compare(baseline, latest)

    try:
        res = int(res)
    except Exception:
        res = "error"

    if res == "error":
        single_gsb_dict["error"] += 1
    else:
        if res > 0:
            single_gsb_dict["B"] += 1
        elif res == 0:
            single_gsb_dict["S"] += 1
        elif res < 0:
            single_gsb_dict["G"] += 1
    return single_gsb_dict


def sublayer_perf_gsb_gen(compare_dict, compare_list):
    """
    性能比率gsb生成
    :param compare_dict:{'layer1':
    {'dy2st_eval_cinn_perf^layercase': 0.2333,
    'dy_eval_perf^layercase': 0.114514,
    'dy2st_eval_cinn_perf^dy_eval_perf^compare': '20%'}}
    :return:
    """
    gsb_dict = {}
    for compare in compare_list:
        if compare["baseline"] == "ground_truth":
            gsb_dict[compare["latest"] + "^" + "compare"] = {"G": 0, "S": 0, "B": 0, "error": 0}
        else:
            gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"] = {
                "G": 0,
                "S": 0,
                "B": 0,
                "error": 0,
            }

    for layer_name, perf_dict in compare_dict.items():
        for compare in compare_list:
            if compare["baseline"] == "ground_truth":
                single_gsb_dict = gsb_ratio_rule(
                    res=perf_dict[compare["latest"] + "^" + "compare"],
                    single_gsb_dict=gsb_dict[compare["latest"] + "^" + "compare"],
                )
                gsb_dict[compare["latest"] + "^" + "compare"] = single_gsb_dict
            else:
                single_gsb_dict = gsb_ratio_rule(
                    res=perf_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"],
                    single_gsb_dict=gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"],
                )
                gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"] = single_gsb_dict

    for key, value in gsb_dict.items():
        all_num = value["G"] + value["S"] + value["B"]
        gsb_dict[key]["G_ratio"] = f"{round(value['G'] / all_num * 100, 2)}%"
        gsb_dict[key]["S_ratio"] = f"{round(value['S'] / all_num * 100, 2)}%"
        gsb_dict[key]["B_ratio"] = f"{round(value['B'] / all_num * 100, 2)}%"

    return gsb_dict


def kernel_perf_gsb_gen(compare_dict, compare_list):
    """
    性能比率gsb生成
    :param compare_dict:{'layer1':
    {'dy2st_eval_cinn_perf-kernel_time^layercase': 2101582.2,
    'dy2st_eval_cinn_perf-kernel_count^layercase': 115,
    'dy_eval_perf-kernel_time^layercase': 11624988.3,
    'dy_eval_perf-kernel_count^layercase': 1765,
    'dy2st_eval_cinn_perf^dy_eval_perf^kernel_time_compare': '453.15%'}}
    :return:
    """
    gsb_dict = {}
    for compare in compare_list:
        if compare["baseline"] == "ground_truth":
            gsb_dict[compare["latest"] + "^" + "kernel_time_compare"] = {"G": 0, "S": 0, "B": 0, "error": 0}
            gsb_dict[compare["latest"] + "^" + "kernel_count_compare"] = {"G": 0, "S": 0, "B": 0, "error": 0}
        else:
            gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_time_compare"] = {
                "G": 0,
                "S": 0,
                "B": 0,
                "error": 0,
            }
            gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_count_compare"] = {
                "G": 0,
                "S": 0,
                "B": 0,
                "error": 0,
            }

    for layer_name, perf_dict in compare_dict.items():
        for compare in compare_list:
            if compare["baseline"] == "ground_truth":
                single_gsb_dict = gsb_ratio_rule(
                    res=perf_dict[compare["latest"] + "^" + "kernel_time_compare"],
                    single_gsb_dict=gsb_dict[compare["latest"] + "^" + "kernel_time_compare"],
                )
                gsb_dict[compare["latest"] + "^" + "kernel_time_compare"] = single_gsb_dict

                if isinstance(perf_dict[compare["latest"] + "-kernel_count^layercase"], str) or isinstance(
                    perf_dict[compare["baseline"] + "-kernel_count^layercase^baseline"], str
                ):
                    count_res = "error"
                else:
                    count_res = (
                        perf_dict[compare["latest"] + "-kernel_count^layercase"]
                        - perf_dict[compare["baseline"] + "-kernel_count^layercase^baseline"]
                    )

                single_gsb_dict = gsb_count_rule(
                    res=count_res,
                    single_gsb_dict=gsb_dict[
                        compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_count_compare"
                    ],
                )
                gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_count_compare"] = single_gsb_dict
            else:
                single_gsb_dict = gsb_ratio_rule(
                    res=perf_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_time_compare"],
                    single_gsb_dict=gsb_dict[
                        compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_time_compare"
                    ],
                )
                gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_time_compare"] = single_gsb_dict

                if isinstance(perf_dict[compare["latest"] + "-kernel_count^layercase"], str) or isinstance(
                    perf_dict[compare["baseline"] + "-kernel_count^layercase"], str
                ):
                    count_res = "error"
                else:
                    count_res = (
                        perf_dict[compare["latest"] + "-kernel_count^layercase"]
                        - perf_dict[compare["baseline"] + "-kernel_count^layercase"]
                    )

                single_gsb_dict = gsb_count_rule(
                    res=count_res,
                    single_gsb_dict=gsb_dict[
                        compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_count_compare"
                    ],
                )
                gsb_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "kernel_count_compare"] = single_gsb_dict

    for key, value in gsb_dict.items():
        all_num = value["G"] + value["S"] + value["B"]
        gsb_dict[key]["G_ratio"] = f"{round(value['G'] / all_num * 100, 2)}%"
        gsb_dict[key]["S_ratio"] = f"{round(value['S'] / all_num * 100, 2)}%"
        gsb_dict[key]["B_ratio"] = f"{round(value['B'] / all_num * 100, 2)}%"

    return gsb_dict


def sublayer_perf_ratio_gen(compare_dict, compare_list):
    """
    平均性能提升, 计算公式: mean(layer1_perf_engine1/layer1_perf_engine2 + layer2_perf_engine1/layer2_perf_engine2...)
    :param compare_dict:{'layer1':
    {'dy2st_eval_cinn_perf^layercase': 0.2333,
    'dy_eval_perf^layercase': 0.114514,
    'dy2st_eval_cinn_perf^dy_eval_perf^compare': '20%'}}
    :return: dict
    """
    ratio_dict = {}
    for compare in compare_list:
        if compare["baseline"] == "ground_truth":
            ratio_dict[compare["latest"] + "^" + "compare"] = {"add_ratio": 0, "num_count": 0}
        else:
            ratio_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"] = {
                "add_ratio": 0,
                "num_count": 0,
            }

    for layer_name, perf_dict in compare_dict.items():
        for compare in compare_list:
            if compare["baseline"] == "ground_truth":
                if perf_dict[compare["latest"] + "^" + "compare"] == "None":
                    continue
                ratio_dict[compare["latest"] + "^" + "compare"]["add_ratio"] = (
                    ratio_dict[compare["latest"] + "^" + "compare"]["add_ratio"]
                    + float(perf_dict[compare["latest"] + "^" + "compare"].strip("%")) / 100
                )
                ratio_dict[compare["latest"] + "^" + "compare"]["num_count"] += 1

            else:
                if perf_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"] == "None":
                    continue
                ratio_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"]["add_ratio"] = (
                    ratio_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"]["add_ratio"]
                    + float(perf_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"].strip("%")) / 100
                )
                ratio_dict[compare["latest"] + "^" + compare["baseline"] + "^" + "compare"]["num_count"] += 1

    for key, value in ratio_dict.items():
        ratio_dict[key][
            "mean_ratio"
        ] = f"{round(ratio_dict[key]['add_ratio'] / ratio_dict[key]['num_count'] * 100, 2)}%"

    return ratio_dict


# def gsb_gen(sublayer_dict, compare_list):
#     """
#     等级分类
#     :param sublayer_dict:{'layer1': {'dy_eval': 0.2333, 'dy2st_eval': 0.114514}}
#     :return:
#     """
#     for layer_name, perf_dict in sublayer_dict.items():
#         for compare in compare_list:
#             if compare["baseline"] == "ground_truth": # gsb不统计上一次vs最新
#                 continue


# for metric_name, metric_value in perf_dict.items():
#     if metric_name == "ground_truth":
#         continue


# grade_dict = {}
# grade_dict["worse"] = []
# grade_dict["doubt"] = []
# grade_dict["equal"] = []
# grade_dict["better"] = []

# for case_name, compare_dict in compare_res.items():
#     tmp = {}
#     # grade = gsb_ratio_rule(res=compare_dict["forward"])
#     # tmp[compare_dict["latest_api"]] = compare_dict["forward"]
#     grade = gsb_ratio_rule(res=compare_dict["best_total"])
#     tmp[compare_dict["latest_api"]] = compare_dict["best_total"]
#     grade_dict[grade].append(tmp)

# return grade_dict
