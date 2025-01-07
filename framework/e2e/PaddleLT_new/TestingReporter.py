#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
与Pisector二分定位模块交互产出测试报告
"""

import os
import json
from pltools.res_save import xlsx_save
from pltools.logger import Logger
from pltools.upload_bos import UploadBos
from db.layer_db import LayerBenchmarkDB
from db.info_map import precision_md5, precision_flags, performance_md5

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
        self.logger = Logger("PLTReporter")
        self.logger.get_log().info(f"task list: {task_list}")
        self.logger.get_log().info(f"date interval: {date_interval}")
        if "," in date_interval:
            date_interval = date_interval.split(",")
        self.date_interval = date_interval
        self.logger.get_log().info(f"self.date_interval: {self.date_interval}")

        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)

    def get_fail_case_info(self):
        """
        获取失败case信息
        """
        layer_db = LayerBenchmarkDB(storage=self.storage)
        relative_fail_dict, absolute_fail_dict = layer_db.get_precision_fail_case_dict(
            task_list=self.task_list, date_interval=self.date_interval
        )
        xlsx_save(relative_fail_dict, "./relative_fail_dict.xlsx")
        xlsx_save(absolute_fail_dict, "./absolute_fail_dict.xlsx")
        return relative_fail_dict, absolute_fail_dict

    def get_fail_case_num(self, fail_dict):
        """
        获取失败case信息
        """
        # layer_db = LayerBenchmarkDB(storage=self.storage)
        # relative_fail_dict, absolute_fail_dict = layer_db.get_precision_fail_case_dict(
        #     task_list=self.task_list, date_interval=self.date_interval
        # )

        fail_num_dict = {}
        for task, value_dict in fail_dict.items():
            fail_num_dict[task] = len(value_dict["fail_list"])
        return fail_num_dict

    def _set_flags(self, task):
        """
        设定环境变量
        """
        task_flags_dict = precision_flags[task]
        for key, value in task_flags_dict.items():
            os.environ[key] = value
            self.logger.get_log().info(f"_set_flags设定环境变量: {key}={value}")

    def _unset_flags(self, task):
        """
        取消环境变量
        """
        task_flags_dict = precision_flags[task]
        for key, value in task_flags_dict.items():
            if key in os.environ:
                self.logger.get_log().info(f"_unset_flags取消环境变量: {key}={os.environ[key]}")
                del os.environ[key]

    def binary_search(self, fail_dict, loop_num=1):
        """
        使用二分工具
        """
        # relative_fail_dict, absolute_fail_dict = self.get_fail_case_info()
        res_dict = {}
        fail_info_dict = {}
        for task, value_dict in fail_dict.items():
            fail_info_dict[task] = value_dict
            fail_info_dict[task]["fail_info"] = {}
            res_dict[task] = {}
            # 设定环境变量
            self._set_flags(task=task)
            if len(value_dict["fail_list"]) == 0:
                self.logger.get_log().info(f"{task}任务无报错case, 无需进行二分定位")
                continue
            else:
                self.logger.get_log().info(f"{task}任务有报错case, 准备进行二分定位")
                baseline_commit = value_dict["baseline_commit"]
                latest_commit = value_dict["latest_commit"]
                testing = value_dict["testing"]
                for layer_file in value_dict["fail_list"]:
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
                    fail_info_dict[task]["fail_info"].update(
                        {
                            layer_file: {
                                "final_commit": final_commit,
                                "check_info": check_info,
                            }
                        }
                    )
            # 取消环境变量
            self._unset_flags(task=task)

        xlsx_save(fail_info_dict, "./binary_search_result.xlsx")
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

    def _bs_upload(self):
        """二分结果上传bos"""
        bos_path = f"PaddleLT/binary_search/build_{self.AGILE_PIPELINE_BUILD_ID}"
        excel_file = "binary_search_result.xlsx"
        if os.path.exists(excel_file):
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path=excel_file)
            self.logger.get_log().info("表格下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, excel_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date_interval", type=str, default="None", help="时间区间选择")
    parser.add_argument("--loop_num", type=int, default=1, help="循环验证次数")
    args = parser.parse_args()
    reporter = TestingReporter(date_interval=args.date_interval)  # date_interval=2024-11-13,2024-11-14
    # 打印出相对失败case信息
    relative_fail_dict, absolute_fail_dict = reporter.get_fail_case_info()
    # print(f"relative_fail_dict:{relative_fail_dict}")
    relative_fail_num_dict = reporter.get_fail_case_num(fail_dict=relative_fail_dict)
    # print(f"relative_fail_num_dict:{relative_fail_num_dict}")
    absolute_fail_num_dict = reporter.get_fail_case_num(fail_dict=absolute_fail_dict)
    # print(f"absolute_fail_num_dict:{absolute_fail_num_dict}")
    # exit(0)
    # 打印出commit定位结果
    res_dict = reporter.binary_search(fail_dict=relative_fail_dict, loop_num=args.loop_num)
    reporter._bs_upload()
    # print("binary search end")
    # print(f"res_dict:{res_dict}")
