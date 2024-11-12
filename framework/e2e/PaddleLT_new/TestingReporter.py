#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
与Pisector二分定位模块交互产出测试报告
"""

import os
from pltools.res_save import xlsx_save
from pltools.logger import Logger
from db.layer_db import LayerBenchmarkDB
from db.info_map import precision_md5, performance_md5

from binary_search import BinarySearch


class TestingReporter(object):
    """
    TestingReporter 用户模块
    """

    def __init__(self, task_list=list(precision_md5.keys()), date_interval=None):
        """
        init
        """

        self.pwd = os.getcwd()

        self.storage = "./apibm_config.yml"
        self.task_list = task_list
        self.date_interval = date_interval
        self.logger = Logger("PLTReporter")

    def get_fail_case_info(self):
        """
        获取失败case信息
        """
        layer_db = LayerBenchmarkDB(storage=self.storage)
        relative_fail_dict, absolute_fail_dict = layer_db.get_precision_fail_case_dict(
            task_list=self.task_list, date_interval=self.date_interval
        )
        xlsx_save(relative_fail_dict, "./relative_fail_dict.xlsx")
        return relative_fail_dict, absolute_fail_dict

    def get_fail_case_num(self):
        """
        获取失败case信息
        """
        layer_db = LayerBenchmarkDB(storage=self.storage)
        relative_fail_dict, absolute_fail_dict = layer_db.get_precision_fail_case_dict(
            task_list=self.task_list, date_interval=self.date_interval
        )

        absolute_fail_num_dict = {}
        for task, value_dict in absolute_fail_dict.items():
            absolute_fail_num_dict[task] = len(value_dict["absolute_fail_list"])
        return absolute_fail_num_dict

    def binary_search(self, loop_num=1):
        """
        使用二分工具
        """
        relative_fail_dict, absolute_fail_dict = self.get_fail_case_info()
        res_dict = {}
        relative_fail_info_dict = {}
        for task, value_dict in relative_fail_dict.items():
            relative_fail_info_dict[task] = value_dict
            relative_fail_info_dict[task]["relative_fail_info"] = {}
            res_dict[task] = {}
            if len(value_dict["relative_fail_list"]) == 0:
                continue
            else:
                baseline_commit = value_dict["baseline_commit"]
                latest_commit = value_dict["latest_commit"]
                testing = value_dict["testing"]
                for layer_file in value_dict["relative_fail_list"]:
                    bs = BinarySearch(
                        good_commit=baseline_commit,
                        bad_commit=latest_commit,
                        layerfile=layer_file,
                        testing=testing,
                        loop_num=loop_num,
                    )
                    final_commit, commit_list, commit_list_origin, check_info = bs._run()
                    res_dict[task][layer_file] = {
                        "final_commit": final_commit,
                        "commit_list": commit_list,
                        "commit_list_origin": commit_list_origin,
                        "check_info": check_info,
                    }
                    relative_fail_info_dict[task]["relative_fail_info"].update(
                        {
                            layer_file: {
                                "final_commit": final_commit,
                                "check_info": check_info,
                            }
                        }
                    )

        xlsx_save(relative_fail_info_dict, "./relative_fail_info_dict.xlsx")
        return res_dict

    # def binary_search_old(self):
    #     """
    #     使用二分工具
    #     """
    #     self.pisector_root = "./Pisector"
    #     self.pisector_settings_yaml = os.path.join(self.pisector_root, "settings.yaml")
    #     self.pisector_run_sh = os.path.join(self.pisector_root, "workshop", "run.sh")
    #     os.system(f'sed -i "s/^cuda_version: .*/cuda_version: \'linux_cuda11.8\'/g" {self.pisector_settings_yaml}')

    #     relative_fail_dict, absolute_fail_dict = self.get_fail_case_info()
    #     for task, value_dict in relative_fail_dict.items():
    #         if len(value_dict["relative_fail_list"]) == 0:
    #             continue
    #         else:
    #             baseline_commit = value_dict["baseline_commit"]
    #             latest_commit = value_dict["latest_commit"]
    #             testing = value_dict["testing"]
    #             for layer_file in value_dict["relative_fail_list"]:
    #                 pwd_tmp = self.pwd.replace("/", "\/")
    #                 layer_file_tmp = layer_file.replace("/", "\/")
    #                 testing_tmp = testing.replace("/", "\/")
    #                 os.system(f'sed -i "s/^python.*/cd xxx/g" {self.pisector_run_sh}')
    #                 os.system(f'sed -i "s/^cd.*/cd {pwd_tmp} \&\& python layertest.py \
    #                        --layerfile {layer_file_tmp} --testing {testing_tmp}/g" {self.pisector_run_sh}')

    #                 os.system(f'sed -i "s/^bad_commit: .*/bad_commit: \"{latest_commit}\"/g" \
    #                        {self.pisector_settings_yaml}')
    #                 os.system(f'sed -i "s/^good_commit: .*/good_commit: \"{baseline_commit}\"/g" \
    #                        {self.pisector_settings_yaml}')

    #                 pis = Pisector("settings.yaml")
    #                 commit, commit_list, commit_list_origin = pis.main()
    #                 print("commit is: {}".format(commit))
    #                 print("commit list is: {}".format(commit_list))
    #                 print("commit list origin is: {}".format(commit_list_origin))


if __name__ == "__main__":
    # reporter = TestingReporter(date_interval=["2024-11-05", "2024-11-07"])
    reporter = TestingReporter()
    # 打印出相对失败case信息
    relative_fail_dict, absolute_fail_dict = reporter.get_fail_case_info()
    print(f"relative_fail_dict:{relative_fail_dict}")
    absolute_fail_num_dict = reporter.get_fail_case_num()
    print(f"absolute_fail_num_dict:{absolute_fail_num_dict}")
    # exit(0)
    # 打印出commit定位结果
    res_dict = reporter.binary_search(loop_num=10)
    print("test end")
    print(f"res_dict:{res_dict}")
