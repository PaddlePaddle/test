#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
子图db交互模块
"""

import os
import json
import traceback
import socket
import platform
from datetime import datetime
from db.db import DB
from db.snapshot import Snapshot
from db.info_map import precision_md5, performance_md5

# from strategy.compare import perf_compare
from pltools.logger import Logger
from pltools.res_save import xlsx_save


class LayerBenchmarkDB(object):
    """
    layer benchmark 交互模块
    """

    def __init__(self, baseline_comment="baseline_CE_layer_benchmark", storage="apibm_config.yaml"):
        """
        :param storage: 信息配置文件
        """
        self.storage = storage
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # md5唯一标识码
        if os.environ.get("PLT_MD5"):
            self.md5_id = os.environ.get("PLT_MD5")  # 手动设置
        else:
            self.md5_id = Snapshot().get_md5_id()  # 自动获取

        # 效率云环境变量
        self.AGILE_PIPELINE_BUILD_ID = int(os.environ.get("AGILE_PIPELINE_BUILD_ID", 0))

        # 例行标识
        self.baseline_comment = baseline_comment
        self.comment = "layer_benchmark_xly_{}".format(self.AGILE_PIPELINE_BUILD_ID)
        self.ci = 0

        # 框架环境信息
        self.framework = os.environ.get("FRAMEWORK")
        self.hardware = os.environ.get("PLT_SET_DEVICE")
        self._frame_info()

        # 机器系统信息
        self.hostname = socket.gethostname()
        self.system = platform.system()

        # 测试配置
        self.testing = os.environ.get("TESTING")
        self.testing_mode = os.environ.get("TESTING_MODE")
        self.plt_perf_content = os.environ.get("PLT_PERF_CONTENT")

        # 子图种类信息
        self.layer_type = os.environ.get("CASE_TYPE")

        # 初始化日志
        self.logger = Logger("LayerBenchmarkDB")
        # self.logger.get_log().info(f"md5_id is: {self.md5_id}")

    def _frame_info(self):
        """"""
        if os.environ.get("FRAMEWORK") == "paddle":
            import paddle

            self.commit = paddle.__git_commit__
            self.version = paddle.__version__
            if os.environ.get("PLT_SET_DEVICE") == "gpu":
                self.env_info = {
                    "cuda": paddle.version.cuda(),
                    "cudnn": paddle.version.cudnn(),
                    "python": os.environ.get("python_ver"),
                    "wheel_link": os.environ.get("wheel_url"),
                    "docker_type": os.environ.get("docker_type"),
                }
            elif os.environ.get("PLT_SET_DEVICE") == "cpu":
                self.env_info = {
                    "python": os.environ.get("python_ver"),
                    "wheel_link": os.environ.get("wheel_url"),
                    "docker_type": os.environ.get("docker_type"),
                }
            else:
                raise Exception("unknown hardware, PaddleLayerTest only support test cpu or gpu")
        elif os.environ.get("FRAMEWORK") == "torch":
            import torch

            self.commit = torch.__git_commit__
            self.version = torch.__version__

            self.env_info = {}
        else:
            raise Exception("unknown framework, PaddleLayerTest only support test PaddlePaddle or Pytorch")

    def latest_insert(self, data_dict, error_list):
        """
        插入最新数据
        """
        db = DB(storage=self.storage)

        if bool(error_list):
            result = "失败"
        else:
            result = "成功"

        # 插入layer_job
        latest_id = db.insert_job(
            comment=self.comment,
            status="running",
            result=result,
            env_info=json.dumps(self.env_info),
            framework=self.framework,
            agile_pipeline_build_id=self.AGILE_PIPELINE_BUILD_ID,
            testing_mode=self.testing_mode,
            testing=self.testing,
            plt_perf_content=self.plt_perf_content,
            layer_type=self.layer_type,
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            hardware=self.hardware,
            system=self.system,
            md5_id=self.md5_id,
            base=0,  # 非基线任务
            ci=self.ci,
            create_time=self.now_time,
            update_time=self.now_time,
        )
        # 保存job id到txt
        with open("job_id.txt", "w") as file:
            file.write(str(latest_id))
        self.logger.get_log().info("录入最新latest数据的job_id: {}".format(latest_id))

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=latest_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=latest_id, status="done", update_time=self.now_time)
            self.logger.get_log().warn("error cases: {}".format(error_list))
            # raise Exception("something wrong with layer benchmark job id: {} !!".format(latest_id))
        else:
            db.update_job(id=latest_id, status="done", update_time=self.now_time)

    def get_baseline_dict(self):
        """
        获取baseline dict
        """
        # 获取baseline用于对比
        db = DB(storage=self.storage)
        baseline_job = db.select_baseline_job(
            comment=self.baseline_comment,
            testing=self.testing,
            plt_perf_content=self.plt_perf_content,
            base=1,
            ci=self.ci,
            md5_id=self.md5_id,
        )
        baseline_id = baseline_job["id"]
        baseline_layer_type = baseline_job["layer_type"]
        baseline_list = db.select(table="layer_case", condition_list=["jid = {}".format(baseline_id)])
        baseline_dict = {}
        for i in baseline_list:
            baseline_dict[i["case_name"]] = i
        return baseline_dict, baseline_layer_type

    def get_precision_fail_case_dict(self, task_list=list(precision_md5.keys()), date_interval=None):
        """
        获取 精度失败case dict
        task_list: 默认为["paddlelt_eval_cinn", "paddlelt_train_cinn", ...]
        date_interval: 时间区间["2024-11-01", "2024-11-03"]
        """
        db = DB(storage=self.storage)
        relative_fail_dict = {}
        absolute_fail_dict = {}
        for task in task_list:
            md5_id = precision_md5[task]
            self.logger.get_log().info(f"start get task:{task} data ~~~")
            if isinstance(date_interval, list):
                try:
                    condition_dict = {
                        "md5_id": md5_id,
                    }
                    date0_res = db.select_use_date(
                        table="layer_job", date_str=date_interval[0], condition_dict=condition_dict
                    )
                    baseline_job = date0_res[-1]
                    date1_res = db.select_use_date(
                        table="layer_job", date_str=date_interval[1], condition_dict=condition_dict
                    )
                    latest_job = date1_res[-1]
                except Exception as e:
                    self.logger.get_log().error(traceback.format_exc())
                    self.logger.get_log().error(e)
                    continue
            else:
                condition_list = [
                    "md5_id = '{}'".format(md5_id),
                ]
                res = db.select(table="layer_job", condition_list=condition_list)
                baseline_job = res[-2]
                latest_job = res[-1]

            baseline_id = baseline_job["id"]
            latest_id = latest_job["id"]
            baseline_commit = baseline_job["commit"]
            latest_commit = latest_job["commit"]
            baseline_testing = baseline_job["testing"]
            latest_testing = latest_job["testing"]
            assert baseline_testing == latest_testing
            baseline_update_time = baseline_job["update_time"]
            latest_update_time = latest_job["update_time"]
            baseline_list = db.select(table="layer_case", condition_list=["jid = {}".format(baseline_id)])
            latest_list = db.select(table="layer_case", condition_list=["jid = {}".format(latest_id)])
            baseline_dict = {}
            for i in baseline_list:
                file_name = i["case_name"].replace("^", "/") + ".py"
                baseline_dict[file_name] = json.loads(i["result"])[baseline_testing]
            latest_dict = {}
            for i in latest_list:
                file_name = i["case_name"].replace("^", "/") + ".py"
                latest_dict[file_name] = json.loads(i["result"])[latest_testing]

            relative_fail_list = []
            for file_name, result in baseline_dict.items():
                if baseline_dict[file_name] == "pass" and latest_dict[file_name] == "fail":
                    relative_fail_list.append(file_name)

            absolute_fail_list = []
            for file_name, result in latest_dict.items():
                if latest_dict[file_name] == "fail":
                    absolute_fail_list.append(file_name)

            relative_fail_dict[task] = {
                "baseline_commit": baseline_commit,
                "latest_commit": latest_commit,
                "testing": latest_testing,
                "baseline_update_time": baseline_update_time,
                "latest_update_time": latest_update_time,
                "fail_list": relative_fail_list,
            }

            absolute_fail_dict[task] = {
                "baseline_commit": baseline_commit,
                "latest_commit": latest_commit,
                "testing": latest_testing,
                "baseline_update_time": baseline_update_time,
                "latest_update_time": latest_update_time,
                "fail_list": absolute_fail_list,
            }

        return relative_fail_dict, absolute_fail_dict

    def baseline_insert(self, data_dict, error_list):
        """
        插入最新数据
        """
        db = DB(storage=self.storage)

        if bool(error_list):
            result = "失败"
        else:
            result = "成功"

        # 插入layer_job
        basleine_id = db.insert_job(
            comment=self.baseline_comment,
            status="running",
            result=result,
            env_info=json.dumps(self.env_info),
            framework=self.framework,
            agile_pipeline_build_id=self.AGILE_PIPELINE_BUILD_ID,
            testing_mode=self.testing_mode,
            testing=self.testing,
            plt_perf_content=self.plt_perf_content,
            layer_type=self.layer_type,
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            hardware=self.hardware,
            system=self.system,
            md5_id=self.md5_id,
            base=1,  # 基线任务
            ci=self.ci,
            create_time=self.now_time,
            update_time=self.now_time,
        )
        # 保存job id到txt
        with open("job_id.txt", "w") as file:
            file.write(str(basleine_id))
        self.logger.get_log().info("录入最新baseline数据的job_id: {}".format(basleine_id))

        # 插入layer_case
        for title, perf_dict in data_dict.items():
            db.insert_case(jid=basleine_id, case_name=title, result=json.dumps(perf_dict), create_time=self.now_time)

        if bool(error_list):
            db.update_job(id=basleine_id, status="done", update_time=self.now_time)
            self.logger.get_log().warn("error cases: {}".format(error_list))
            # raise Exception("something wrong with layer benchmark job id: {} !!".format(basleine_id))
        else:
            db.update_job(id=basleine_id, status="done", update_time=self.now_time)
