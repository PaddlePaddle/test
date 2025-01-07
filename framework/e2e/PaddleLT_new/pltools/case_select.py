#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nn.Layer配置相关
"""

import os

# import platform
# import time
# import pytest
# import allure
# from pltools.yaml_loader import YamlLoader


class CaseSelect(object):
    """通过指定的nn.Layer的yaml, 选择用于测试的cases"""

    def __init__(self, dirpath, ignore_list):
        """init"""
        # self.cur_path = os.getcwd()
        # self.report_dir = os.path.join(self.cur_path, "report")
        # self.env = env
        # self.repo_list = repo_list
        self.dirpath = dirpath
        self.ignore_list = ignore_list

    def get_yaml_list(self, base_path, yaml_list=[]):
        """递归寻找文件夹内所有的yml文件路径"""
        file_list = os.listdir(base_path)

        for file in file_list:
            yaml_path = os.path.join(base_path, file)

            if os.path.isdir(yaml_path):
                self.get_yaml_list(yaml_path, yaml_list)
            else:
                if not file.endswith(".yml"):
                    continue
                else:
                    yaml_list.append(yaml_path)
        return yaml_list

    def get_py_list(self, base_path, py_list=[]):
        """递归寻找文件夹内所有的子图py文件路径"""
        file_list = os.listdir(base_path)

        for file in file_list:
            py_path = os.path.join(base_path, file)

            if os.path.isdir(py_path):
                self.get_py_list(py_path, py_list)
            else:
                if self.ignore_list:
                    if (
                        not file.endswith(".py")
                        or file.endswith("__init__.py")
                        or file.endswith("utils.py")
                        or py_path in self.ignore_list
                    ):
                        continue
                    else:
                        py_list.append(py_path)
                else:
                    if not file.endswith(".py") or file.endswith("__init__.py") or file.endswith("utils.py"):
                        continue
                    else:
                        py_list.append(py_path)
        return py_list
