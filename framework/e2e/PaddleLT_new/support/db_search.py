#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
获取pr信息
"""


import os
import sys
import argparse
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from db.db import DB
from db.snapshot import Snapshot
from db.info_map import precision_md5, precision_flags, performance_md5
import requests


class DBSearcher(object):
    """
    pr信息获取
    """

    def __init__(self, storage="apibm_config.yaml"):
        """
        init
        """
        self.storage = storage

    def get_prec_job_mun(self):
        """
        获取精度任务信息
        """
        db = DB(storage=self.storage)
        # table = "layer_job"
        # condition_list = ["testing_mode = 'precision'"]
        job_list = db.select(table="layer_job", condition_list=["testing_mode = 'precision'"])
        task_count = len(job_list)

        fail_count = 0
        pass_count = 0
        for task in job_list:
            task_testing = task["testing"]
            case_list = db.select(table="layer_case", condition_list=[f"jid = {task['id']}"])
            for case in case_list:
                case_res = json.loads(case["result"])[task_testing]
                if case_res == "fail":
                    fail_count += 1
                if case_res == "pass":
                    pass_count += 1
        print("task_count is: ", task_count)
        print("fail_count is: ", fail_count)
        print("pass_count is: ", pass_count)
        return task_count, fail_count, pass_count

    def get_prec_job_dict(self, task_dict=precision_md5):
        """
        获取精度任务信息
        """
        db = DB(storage=self.storage)
        job_dict = {}
        for key, md5_id in task_dict.items():
            print(key, md5_id)
            job_dict[key] = {"task_count": 0, "fail_count": 0, "pass_count": 0}
            job_list = db.select(table="layer_job", condition_list=[f"md5_id = '{md5_id}'"])

            for task in job_list:
                job_dict[key]["task_count"] += 1
                task_testing = task["testing"]
                case_list = db.select(table="layer_case", condition_list=[f"jid = {task['id']}"])
                for case in case_list:
                    case_res = json.loads(case["result"])[task_testing]
                    if case_res == "fail":
                        job_dict[key]["fail_count"] += 1
                    if case_res == "pass":
                        job_dict[key]["pass_count"] += 1

        print("job_dict is: ", job_dict)
        return job_dict

    def gen_mum(self):
        """
        获取性能任务信息
        """
        tmp_dict = {
            "paddlelt_eval_cinn": {
                "task_count": 276,
                "fail_count": (811 * 31 + 319 * 30 + 281 * 31 + 157 * 30 + 102 * 31 + 62 * 31 + 36 * 30 + 0 * 31 + 26),
            },
            "paddlelt_train_cinn": {
                "task_count": 215,
                "fail_count": (497 * 31 + 188 * 30 + 135 * 31 + 81 * 31 + 48 * 30 + 15 * 31 + 1027),
            },
            "paddlelt_eval_cinn_inputspec": {
                "task_count": 215,
                "fail_count": (1090 * 31 + 809 * 30 + 460 * 31 + 105 * 31 + 52 * 30 + 2 * 31 + 17),
            },
            "paddlelt_train_cinn_inputspec": {
                "task_count": 215,
                "fail_count": (1090 * 31 + 843 * 30 + 521 * 31 + 142 * 31 + 142 * 30 + 155 * 31 + 1194),
            },
            "paddlelt_train_prim_inputspec": {
                "task_count": 215,
                "fail_count": (440 * 31 + 19 * 30 + 15 * 31 + 15 * 31 + 8 * 30 + 0 * 31 + 0),
            },
            "paddlelt_train_pir_infersymbolic_inputspec": {
                "task_count": 215,
                "fail_count": (450 * 31 + 131 * 30 + 101 * 31 + 98 * 31 + 41 * 30 + 2 * 31 + 20),
            },
            "paddlelt_train_api_dy2stcinn_static_inputspec": {
                "task_count": 215,
                "fail_count": (148 * 31 + 101 * 30 + 101 * 31 + 2011 + 1820 + 1658 + 1428),
            },
            "paddlelt_train_api_dy2stcinn_inputspec": {
                "task_count": 215,
                "fail_count": (201 * 31 + 182 * 30 + 182 * 31 + 2811 + 2520 + 1958 + 1948),
            },
        }
        print(tmp_dict)


if __name__ == "__main__":
    searcher = DBSearcher(storage="storage.yml")
    searcher.gen_mum()
