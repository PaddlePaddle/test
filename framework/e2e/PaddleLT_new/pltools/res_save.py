#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
保存工具
"""

import os
import shutil
import tarfile
import pickle
import requests
import wget
import pandas as pd


def dict_key_fix(sublayer_dict, old_substr, new_substr):
    """
    字典中key改名
    """
    new_dict = {}
    for key, value in sublayer_dict.items():
        new_key = key.replace(old_substr, new_substr)
        new_dict[new_key] = value
    return new_dict


def xlsx_save(sublayer_dict, excel_file):
    """
    子图保存到excel
    """
    sublayer_dict = dict_key_fix(sublayer_dict, "perf_monitor^manual_subgraph^", "")
    data = []

    # 遍历嵌套字典，提取数据并添加到列表中
    for key, sub_dict in sublayer_dict.items():
        row = {"sub_layer": key}
        for subkey, value in sub_dict.items():
            row[subkey] = value
        data.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 将数据写入 Excel 文件
    df.to_excel(excel_file, index=False)


def save_tensor(data, filename):
    """
    保存Tensor数据
    """
    save_name = filename + ".tensor"

    if os.environ.get("FRAMEWORK") == "paddle":
        import paddle

        paddle.save(data, save_name)
    elif os.environ.get("FRAMEWORK") == "torch":
        import torch

        torch.save(data, save_name)


def load_tensor(filename):
    """
    加载Tensor数据
    """
    load_name = filename + ".tensor"

    if os.environ.get("FRAMEWORK") == "paddle":
        import paddle

        data = paddle.load(load_name)
    elif os.environ.get("FRAMEWORK") == "torch":
        import torch

        data = torch.load(load_name)
    return data


# dict 保存 为txt
def save_txt(data, filename):
    """
    保存数据为txt
    """
    save_name = filename + ".txt"
    with open(save_name, "w") as f:
        for key in data.keys():
            f.write("{}: {}\n".format(key, data[key]))


# list 保存/加载 为pickle
def save_pickle(data, filename):
    """
    保存数据为pickle
    """
    save_name = filename + ".pickle"
    with open(save_name, "wb") as f:
        # 使用pickle的dump函数将列表写入文件
        pickle.dump(data, f)


def load_pickle(filename):
    """
    加载pickle文件中的精度
    """
    with open(filename, "rb") as f:
        # 使用pickle的load函数从文件中加载列表
        loaded_data = pickle.load(f)

    return loaded_data


def wget_sth(gt_url):
    """
    下载
    """
    wget.download(gt_url)


def download_sth(gt_url, output_path):

    """
    下载文件到指定路径

    :param gt_url: 文件URL
    :param output_path: 文件保存路径
    """
    with requests.get(gt_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def create_tar_gz(file_path, source_dir):
    """
    创建一个gzip压缩的tar文件(.tar.gz)

    :param file_path: 输出的.tgz文件名
    :param source_dir: 要打包的源目录
    """
    with tarfile.open(file_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def extract_tar_gz(file_path, extract_path="./"):
    """
    解压.tar.gz文件

    :param file_path: .tar.gz文件的路径
    :param extract_path: 解压到的目标路径
    """
    with tarfile.open(file_path, "r:gz") as tar_ref:
        tar_ref.extractall(extract_path)
