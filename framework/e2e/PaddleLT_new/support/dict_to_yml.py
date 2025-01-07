#!/bin/env python
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
字典转换为yaml
"""

import yaml
import json


def dict_to_yaml(filename, data_str):
    """
    dict to yml
    """
    data_dict = json.loads(data_str)

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(data_dict, file, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filename", type=str, default="None", help="导出文件名")
    parser.add_argument("--data_str", type=str, default="None", help="字典字符串")
    args = parser.parse_args()
    dict_to_yaml(args.filename, args.data_str)
