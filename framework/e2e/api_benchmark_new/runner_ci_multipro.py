#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
注意!!!!!!!
修复 self.core_index、self.comment 和 self.wheel_link
"""

import os
import multiprocessing
import socket
import platform

import argparse
import json
import sys
from datetime import datetime

from statistics.statistics import Statistics

# from db.db import DB
from db.ci_db import CIdb
from info.snapshot import Snapshot
from strategy.compare import double_check, bad_check, ci_level_reveal, data_compare
from strategy.transdata import data_list_to_dict
from alarm.alarm import Alarm

import paddle

sys.path.append("..")
from utils.logger import Logger
from runner_base import ApiBenchmarkBASE

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--core_index", type=int, default=2, help="index of cpu core")
parser.add_argument("--yaml", type=str, help="input the yaml path")
parser.add_argument("--python", type=str, default="python3.10", help="input the yaml path")
parser.add_argument("--baseline_whl_link", type=str, default=None, help="only be used to insert baseline data")
args = parser.parse_args()

# p = psutil.Process()
# p.cpu_affinity([args.core_index])

SKIP_DICT = {"Windows": ["fft"], "Darwin": ["fft"], "Linux": []}
INDEX_DICT = {}
SPECIAL = False  # speacial for emergency


class ApiBenchmarkCI(ApiBenchmarkBASE):
    """
    api benchmark 调度CI, 监控cpu+前向, 支持多个机器baseline
    """

    def __init__(self, yaml_path, python):
        super(ApiBenchmarkCI, self).__init__(yaml_path)
        """
        :param baseline: 性能baseline键值对, key为case名, value为性能float
        """
        # 测试控制项
        self.core_index = args.core_index  # 第一个cpu核序号
        self.multiprocess_num = 4  # 并行进程数
        self.loops = 50  # 循环次数
        self.base_times = 1000  # timeit 基础运行时间
        self.default_dtype = "float32"
        self.if_showtime = True
        self.double_check = True
        self.check_iters = 5
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # md5唯一标识码
        self.md5_id = Snapshot().get_md5_id()

        # 效率云环境变量
        self.AGILE_PULL_ID = os.environ.get("AGILE_PULL_ID", "0")
        self.AGILE_REVISION = os.environ.get("AGILE_REVISION", "0")
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.description = {"success": True, "reason": "ok", "pipelineBuildId": self.AGILE_PIPELINE_BUILD_ID}

        # 例行标识
        self.baseline_comment = "baseline_CI_api_benchmark_pr_dev"
        self.comment = "CI_api_benchmark_pr_{}_ver_{}".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        # self.comment = "multipro_naive_test"
        self.routine = 0
        self.ci = 1
        self.uid = -1

        # 框架信息
        self.framework = "paddle"
        self.wheel_link = (
            "https://xly-devops.bj.bcebos.com/PR/build_whl/{}/{}"
            "/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl".format(self.AGILE_PULL_ID, self.AGILE_REVISION)
        )

        # self.wheel_link = (
        #     "https://xly-devops.bj.bcebos.com/PR/build_whl/0/8b2b95305d14d57845e9b403da0ec4bf29e45e27"
        #     "/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"
        # )

        # 框架信息callback
        self.commit = paddle.__git_commit__
        self.version = paddle.__version__
        self.cuda = paddle.version.cuda()
        self.cudnn = paddle.version.cudnn()

        # 项目配置信息
        self.place = "cpu"
        self.python = python
        self.enable_backward = 0
        self.yaml_info = "case_0"
        self.card = 0

        # 机器系统信息
        self.hostname = socket.gethostname()
        self.system = platform.system()
        self.snapshot = {
            "os": platform.platform(),
            "card": self.card,
            "cuda": self.cuda,
            "cudnn": self.cudnn,
            "comment": self.comment,
        }

        # 初始化日志
        self.logger = Logger("ApiBenchmarkCI")

        # 初始化统计模块
        self.statistics = Statistics()

        # 邮件报警
        # self.email = Alarm(storage=self.storage)

    # def split_list(self, lst, n):
    #     """
    #     将列表均分为n份
    #     Args:
    #         lst (list): 待划分的列表
    #         n (int): 划分的份数
    #     Returns:
    #         res (list): 划分后的列表，其中每个元素为原列表的1/n部分
    #     """
    #     if not isinstance(lst, list) or not isinstance(n, int) or len(lst) == 0 or n <= 0:
    #         return []

    #     quotient, remainder = divmod(len(lst), n)
    #     res = []
    #     start = 0
    #     for i in range(n):
    #         if i < remainder:
    #             end = start + quotient + 1
    #         else:
    #             end = start + quotient
    #         res.append(lst[start:end])
    #         start = end
    #     return res

    def split_list(self, lst, n):
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

    def _multi_run_main(self, all_cases, loops, base_times, result_queue):
        """
        multi run main
        """
        error_dict = self._run_main(all_cases=all_cases, loops=loops, base_times=base_times)
        result_queue.put(error_dict)

    def _run_ci(self):
        """

        :return:
        """
        multiprocess_cases = self.split_list(lst=list(self.all_cases), n=self.multiprocess_num)
        processes = []
        result_queue = multiprocessing.Queue()

        for i, cases_list in enumerate(multiprocess_cases):
            process = multiprocessing.Process(
                target=self._multi_run_main, args=(cases_list, self.loops, self.base_times, result_queue)
            )
            process.start()
            os.sched_setaffinity(process.pid, {self.core_index + i})
            processes.append(process)

        for process in processes:
            process.join()

        error_dict = {}
        while not result_queue.empty():
            result_dict = result_queue.get()
            error_dict.update(result_dict)

        # error_dict = self._run_main(all_cases=self.all_cases, loops=self.loops, base_times=self.base_times)

        # 查询数据库构建baseline
        db = CIdb(storage=self.storage)
        baseline_id = db.ci_select_baseline_job(
            comment=self.baseline_comment, routine=1, ci=self.ci, md5_id=self.md5_id
        )
        baseline_list = db.select(table="case", condition_list=["jid = {}".format(baseline_id)])

        baseline_dict = data_list_to_dict(baseline_list)

        latest_id = db.ci_insert_job(
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            place=self.place,
            system=self.system,
            cuda=self.cuda,
            cudnn=self.cudnn,
            snapshot=json.dumps(self.snapshot),
            md5_id=self.md5_id,
            uid=self.uid,
            routine=self.routine,
            ci=self.ci,
            comment=self.comment,
            enable_backward=self.enable_backward,
            python=self.python,
            yaml_info=self.yaml_info,
            wheel_link=self.wheel_link,
            description=json.dumps(self.description),
            create_time=self.now_time,
            update_time=self.now_time,
        )

        ci_dict = {}
        data = dict()
        for i in os.listdir("./log/"):
            with open("./log/" + i) as case:
                res = case.readline()
                api = i.split(".")[0]
                data[api] = res
        for k, v in data.items():
            ci_dict[k] = {}
            ci_dict[k]["case_name"] = k
            ci_dict[k]["api"] = json.loads(v).get("api")
            ci_dict[k]["result"] = v

        compare_dict = {}
        bad_check_case = []
        double_check_case = []
        for k, v in ci_dict.items():
            baseline_case = baseline_dict[k]
            latest_case = ci_dict[k]
            compare_res = data_compare(baseline_case=baseline_case, latest_case=latest_case, case_name=k)
            compare_dict[k] = compare_res[k]
            if bad_check(res=compare_res[k]):
                bad_check_case.append(k)
            if self.double_check and double_check(res=compare_res[k]):
                double_check_case.append(k)

        if self.double_check and bool(double_check_case):
            double_error_dict = self._run_main(
                all_cases=double_check_case, loops=self.loops * 6, base_times=self.base_times
            )
            ci_dict = {}
            data = dict()
            for i in os.listdir("./log/"):
                with open("./log/" + i) as case:
                    res = case.readline()
                    api = i.split(".")[0]
                    data[api] = res
            for k, v in data.items():
                ci_dict[k] = {}
                ci_dict[k]["case_name"] = k
                ci_dict[k]["api"] = json.loads(v).get("api")
                ci_dict[k]["result"] = v

            compare_dict = {}
            for k, v in ci_dict.items():
                baseline_case = baseline_dict[k]
                latest_case = ci_dict[k]
                compare_res = data_compare(baseline_case=baseline_case, latest_case=latest_case, case_name=k)
                compare_dict[k] = compare_res[k]
            print("double_error_dict is: ", double_error_dict)
        else:
            double_error_dict = {}
            print("double_error_dict is: ", double_error_dict)

        self._db_save(db=db, latest_id=latest_id)

        if bool(error_dict):
            db.ci_update_job(id=latest_id, status="error", update_time=self.now_time)
            raise Exception("something wrong with api benchmark CI job id: {} !!".format(latest_id))
        else:
            db.ci_update_job(id=latest_id, status="done", update_time=self.now_time)

        api_grade = ci_level_reveal(compare_dict)
        del api_grade["equal"]
        del api_grade["better"]
        print("double_check_case are: ", double_check_case)
        print(
            "以下为pr{}引入之后，api调度性能相对于baseline的变化。worse表示性能下降超过30%的api，doubt表示性能下降为15%~30%之间的api".format(
                self.AGILE_PULL_ID
            )
        )
        print(api_grade)

        if bool(bad_check_case):
            baseline_whl = db.select_by_id(table="job", id=baseline_id)
            latest_whl = db.select_by_id(table="job", id=latest_id)
            print("性能基线paddle wheel包: {}".format(baseline_whl[-1]["wheel_link"]))
            print("此次测试paddle wheel包: {}".format(latest_whl[-1]["wheel_link"]))
            print("报错case的复现代码链接如下, 请依次安装基线wheel包和测试wheel包, 使用复现代码进行性能对比验证: ")
            for i in bad_check_case:
                print(
                    "https://github.com/PaddlePaddle/PaddleTest/tree/develop/"
                    "framework/e2e/api_benchmark_new/debug_case/{}.py".format(i)
                )

        print(
            "详情差异请点击以下链接查询: http://paddletest.baidu-int.com:8081/#/paddle/benchmark/apiBenchmark/report/{}&{}".format(
                latest_id, baseline_id
            )
        )

        if bool(api_grade["doubt"]) or bool(api_grade["worse"]):
            raise Exception("该pr会导致动态图cpu前向调度性能下降，请修复！！！")

    def _baseline_insert(self, wheel_link):
        """

        :return:
        """
        multiprocess_cases = self.split_list(lst=list(self.all_cases), n=self.multiprocess_num)
        processes = []
        result_queue = multiprocessing.Queue()

        for i, cases_list in enumerate(multiprocess_cases):
            process = multiprocessing.Process(
                target=self._multi_run_main, args=(cases_list, self.loops, self.base_times, result_queue)
            )
            process.start()
            os.sched_setaffinity(process.pid, {self.core_index + i})
            processes.append(process)

        for process in processes:
            process.join()

        error_dict = {}
        while not result_queue.empty():
            result_dict = result_queue.get()
            error_dict.update(result_dict)

        # error_dict = self._run_main(all_cases=self.all_cases, loops=self.loops, base_times=self.base_times)

        # 初始化数据库
        db = CIdb(storage=self.storage)

        job_id = db.ci_insert_job(
            commit=self.commit,
            version=self.version,
            hostname=self.hostname,
            place=self.place,
            system=self.system,
            cuda=self.cuda,
            cudnn=self.cudnn,
            snapshot=json.dumps(self.snapshot),
            md5_id=self.md5_id,
            uid=self.uid,
            routine=1,  # 基线例行标签
            ci=self.ci,
            comment=self.baseline_comment,  # 基线comment
            enable_backward=self.enable_backward,
            python=self.python,
            yaml_info=self.yaml_info,
            # wheel_link=self.wheel_link,
            wheel_link=wheel_link,
            description=json.dumps(self.description),
            create_time=self.now_time,
            update_time=self.now_time,
        )
        # cases_dict, error_dict = self._run_main(
        #     all_cases=self.all_cases, latest_id=job_id, iters=1, compare_switch=False
        # )

        self._db_save(db=db, latest_id=job_id)

        if bool(error_dict):
            db.ci_update_job(id=job_id, status="error", update_time=self.now_time)
            print("error cases: {}".format(error_dict))
            raise Exception("something wrong with api benchmark job id: {} !!".format(job_id))
        else:
            db.ci_update_job(id=job_id, status="done", update_time=self.now_time)

        # # debug
        # cases_dict = self._run_main(all_cases=self.all_cases, latest_id=job_id)
        # self.db.ci_update_job(id=job_id, status="done", update_time=self.now_time)
        # del cases_dict


if __name__ == "__main__":
    # api_bm = ApiBenchmarkCI(yaml_path="./../yaml/api_benchmark_fp32.yml")
    api_bm = ApiBenchmarkCI(yaml_path=args.yaml, python=args.python)
    if bool(args.baseline_whl_link):
        api_bm._baseline_insert(wheel_link=args.baseline_whl_link)
    else:
        api_bm._run_ci()
