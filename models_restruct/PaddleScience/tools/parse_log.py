"""
自定义解析入口
"""
# encoding: utf-8
import re
import ast
import json
import logging
import numpy as np
import sys
logger = logging.getLogger("ce")


def paddlelas_imagenet_parse(log_content, kpi_name):
    # 定义一个空列表来存储提取到的KPI值
    kpi_value_all = []

    # 打开日志文件
    with open(log_content, 'r') as f:
        # 逐行读取日志内容
        for line in f.readlines():
            # 使用正则表达式匹配KPI名称和对应的数值，将提取到的数值添加到kpi_value_all中
            if kpi_name + ":" in line and "[Train]" in line:
                regexp = r"%s:\s*([0-9.]+)" % re.escape(kpi_name)  # 修改正则表达式以支持带有"."的KPI名称
                r = re.findall(regexp, line)
                if len(r) > 0:
                    kpi_value_all.append(float(r[0].strip()))
    
    # 如果没有提取到任何KPI值，则提取eval阶段的指标
    if len(kpi_value_all) == 0:
        # 打开日志文件
        with open(log_content, 'r') as f:
            # 逐行读取日志内容
            for line in f.readlines():
                # 使用正则表达式匹配KPI名称和对应的数值，将提取到的数值添加到kpi_value_all中
                if kpi_name + ":" in line and "[Eval]" in line:
                    regexp = r"%s:\s*([0-9.]+)" % re.escape(kpi_name)  # 修改正则表达式以支持带有"."的KPI名称
                    r = re.findall(regexp, line)
                    if len(r) > 0:
                        kpi_value_all.append(float(r[0].strip()))
    if len(kpi_value_all) == 0:
        kpi_value = sys.maxsize
    else:    
        kpi_value = kpi_value_all[-1]

    # 返回最终的KPI值
    return kpi_value


if __name__ == "__main__":
    log_content = "/ssd2/sjx/sjx_cuda11.8_py310/PaddleScience_test/PaddleScience/test_0618.log"
    kpi_name = "loss"
    print(paddlelas_imagenet_parse(log_content, kpi_name))
