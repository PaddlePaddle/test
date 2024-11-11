#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
二分工具
"""
import os
import ast
import sys
import argparse
import subprocess
import requests
import numpy as np
from layertest import LayerTest
from strategy.compare import perf_compare
from pltools.logger import Logger
from pltools.res_save import save_pickle


# def get_commits(start, end):
#     """
#     get all the commits in search interval
#     """
#     print("start:{}".format(start))
#     print("end:{}".format(end))
#     cmd = "git log {}..{} --pretty=oneline".format(start, end)
#     log = subprocess.getstatusoutput(cmd)
#     print(log[1])
#     commit_list = []
#     candidate_commit = log[1].split("\n")
#     print(candidate_commit)
#     for commit in candidate_commit:
#         commit = commit.split(" ")[0]
#         print("commit:{}".format(commit))
#         commit_list.append(commit)
#     return commit_list


class BinarySearch(object):
    """
    性能/精度通用二分定位工具
    """

    def __init__(self, good_commit, bad_commit, layerfile, testing, perf_decay=None, test_obj=LayerTest):
        """
        初始化
        good_commit: pass的commit
        bad_commit: fail的commit
        layerfile: 子图路径, 例如./layercase/sublayer1000/Det_cases/ppyolo_ppyolov2_r50vd_dcn_365e_coco/SIR_76.py
        testing: 测试yaml路径, 例如 yaml/dy^dy2stcinn_eval_benchmark.yml
        perf_decay: 仅用于性能, 某个engine名称+预期耗时+性能下降比例, 组成的list, 例如["dy2st_eval_cinn_perf", 0.0635672, -0.3]

        """
        self.logger = Logger("PLT二分定位")

        self.cur_path = os.getcwd()
        if not os.path.exists("Paddle-develop"):
            os.system(
                "wget -q https://xly-devops.bj.bcebos.com/PaddleTest/Paddle/Paddle-develop.tar.gz \
                && tar -xzf Paddle-develop.tar.gz"
            )

        self.good_commit = good_commit
        self.bad_commit = bad_commit
        self.whl_link_template = (
            "https://paddle-qa.bj.bcebos.com/paddle-pipeline/"
            "Develop-GpuSome-LinuxCentos-Gcc82-Cuda118-Cudnn86-Trt85-Py310-CINN-Compile/{}/paddle"
            "paddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl"
        )

        self.layerfile = layerfile
        self.title = self.layerfile.replace(".py", "").replace("/", "^").replace(".", "^")
        self.testing = testing
        self.perf_decay = perf_decay
        self.test_obj = test_obj
        self.py_cmd = os.environ.get("python_ver")
        self.testing_mode = os.environ.get("TESTING_MODE")
        self.device_place_id = 0
        self.timeout = 300

    def _get_commits(self):
        """
        get all the commits in search interval
        """

        self.logger.get_log().info(f"good_commit: {self.good_commit}")
        self.logger.get_log().info(f"bad_commit: {self.bad_commit}")

        os.chdir(os.path.join(self.cur_path, "Paddle-develop"))
        cmd = "git log {}..{} --pretty=oneline".format(self.good_commit, self.bad_commit)
        log = subprocess.getstatusoutput(cmd)
        # self.logger.get_log().info(log[1])
        os.chdir(self.cur_path)

        commit_list = []
        candidate_commit = log[1].split("\n")
        # self.logger.get_log().info(candidate_commit)
        for commit in candidate_commit:
            commit = commit.split(" ")[0]
            # self.logger.get_log().info("commit:{}".format(commit))
            commit_list.append(commit)
        return commit_list

    def _check_downloadable(self, url):
        """
        # 发送 HEAD 请求，检查响应头信息
        """
        try:
            # 发送 HEAD 请求，检查响应头信息
            self.logger.get_log().info(f"Checking if the file is downloadable: {url}")
            response = requests.head(url, allow_redirects=True)
            if response.status_code == 200:
                # 获取文件大小
                file_size = response.headers.get("Content-Length", None)
                if file_size:
                    self.logger.get_log().info(f"File size: {int(file_size) / (1024 * 1024):.2f} MB")
                else:
                    self.logger.get_log().error("File size could not be determined.")
                return True
            else:
                self.logger.get_log().error(f"Failed to access the file. Status code: {response.status_code}")
                return False
        except Exception as e:
            self.logger.get_log().error(f"An error occurred: {e}")
            return False

    def _check_package_available(self, commit_list):
        """
        检查全部包是否存在
        """
        available_commits = []
        available_commits_dict = {}
        for commit in commit_list:
            if not self._check_downloadable(self.whl_link_template.replace("{}", commit)):
                self.logger.get_log().info(f"===> 【{commit}】安装包不存在 <===")
                available_commits_dict[commit] = False
            else:
                available_commits_dict[commit] = True
                available_commits.append(commit)
        if len(available_commits) < len(commit_list):
            self.logger.get_log().warning("===> 部分commit list的安装包不可用, 使用现有可用包Commit列表, 结果仅供参考。 <===")
            self.logger.get_log().info("===> 检查相关commit list的安装包可用情况如下: <===")
            for k, v in available_commits_dict.items():
                self.logger.get_log().info(f"===>  【{k}】安装包可用情况 {v} <===")
        else:
            self.logger.get_log().info("===> 相关commit list的安装包全部可用 <===")
        return available_commits

    def _status_print(self, exit_code, status_str):
        """
        状态打印
        """
        if exit_code == 0:
            self.logger.get_log().info(f"{status_str} successfully")
        else:
            self.logger.get_log().error(f"{status_str} failed")
            sys.exit(-1)

    def _install_paddle(self, commit_id):
        """
        安装 paddle
        """
        exit_code = os.system(f"{self.py_cmd} -m pip uninstall paddlepaddle-gpu -y")
        self._status_print(exit_code=exit_code, status_str="uninstall paddlepaddle-gpu")

        whl_link = self.whl_link_template.replace("{}", commit_id)

        exit_code = os.system(f"{self.py_cmd} -m pip install {whl_link}")
        self._status_print(exit_code=exit_code, status_str="install paddlepaddle-gpu")
        self.logger.get_log().info("commit {} install done".format(commit_id))
        return 0

    def _precision_debug(self, commit_id):
        """
        精度debug
        """
        # exc = 0
        # try:
        #     self.test_obj(title=self.title, layerfile=self.layerfile, testing=self.testing)._case_run()
        # except Exception:
        #     exc += 1

        if os.path.exists(f"{self.title}.py"):
            os.system(f"rm {self.title}.py")
        exit_code = os.system(
            f"cp -r PaddleLT.py {self.title}.py && "
            f"{self.py_cmd} -m pytest {self.title}.py --title={self.title} "
            f"--layerfile={self.layerfile} --testing={self.testing} "
            f"--device_place_id={self.device_place_id} --timeout={self.timeout}"
        )

        if exit_code > 0:
            self.logger.get_log().info(f"{self.testing_mode}执行失败commit: {commit_id}")
            return False
        else:
            self.logger.get_log().info(f"{self.testing_mode}执行成功commit: {commit_id}")
            return True

    def _performance_debug(self, commit_id):
        """
        精度debug
        """
        res_dict, exc = self.test_obj(title=self.title, layerfile=self.layerfile, testing=self.testing)._perf_case_run()
        latest = res_dict[self.perf_decay[0]]
        baseline = self.perf_decay[1]
        decay_rate = self.perf_decay[2]

        compare_res = perf_compare(baseline, latest)
        fluctuate_rate = 0.15
        if exc > 0 or compare_res < decay_rate - fluctuate_rate:
            self.logger.get_log().info(f"{self.testing_mode}执行失败commit: {commit_id}")
            return False
        else:
            self.logger.get_log().info(f"{self.testing_mode}执行成功commit: {commit_id}")
            return True

    def _commit_locate(self, commits):
        """
        commit定位
        """
        self.logger.get_log().info("测试case名称: {}".format(self.title))

        if len(commits) == 2:
            self.logger.get_log().info(
                "only two candidate commits left in binary_search, the final commit is {}".format(commits[0])
            )
            return commits[0]
        left, right = 0, len(commits) - 1
        if left <= right:
            mid = left + (right - left) // 2
            commit = commits[mid]

            self._install_paddle(commit)

            if eval(f"self._{self.testing_mode}_debug")(commit):
                self.logger.get_log().info("the commit {} is success".format(commit))
                self.logger.get_log().info("mid value:{}".format(mid))
                selected_commits = commits[: mid + 1]
                res = self._commit_locate(selected_commits)
            else:
                self.logger.get_log().info("the commit {} is failed".format(commit))
                selected_commits = commits[mid:]
                res = self._commit_locate(selected_commits)
        return res

    def _run(self):
        """
        用户运行
        """
        commit_list_origin = self._get_commits()
        self.logger.get_log().info(f"original commit list is: {commit_list_origin}")
        save_pickle(data=commit_list_origin, filename="commit_list_origin.pickle")

        commit_list = self._check_package_available(commit_list=commit_list_origin)
        self.logger.get_log().info(f"real commit list is: {commit_list}")
        save_pickle(data=commit_list, filename="commit_list.pickle")

        final_commit = self._commit_locate(commits=commit_list)

        return final_commit, commit_list, commit_list_origin


if __name__ == "__main__":
    # cur_path = os.getcwd()
    # if not os.path.exists("paddle"):
    #     os.system("git clone -b develop http://github.com/paddlepaddle/paddle.git")
    # os.chdir(os.path.join(cur_path, "paddle"))
    # start_commit = "651e66ba06f3ae26c3cf649f83a9a54b486ce75d"  # 成功commit
    # end_commit = "5ad596c983e6f6626a6d879b17834d15664946bc"  # 失败commit
    # commits = get_commits(start=start_commit, end=end_commit)
    # save_pickle(data=commits, filename="candidate_commits.pickle")
    # print("the candidate commits is {}".format(commits))

    # os.chdir(cur_path)
    # final_commit = BinarySearch(
    #     commit_list=commits,
    #     title="PrecisionBS",
    #     layerfile="./layercase/sublayer1000/Det_cases/gfl_gfl_r101vd_fpn_mstrain_2x_coco/SIR_145.py",
    #     testing="yaml/dy^dy2stcinn_train_inputspec.yml",
    #     perf_decay=None,  # ["dy2st_eval_cinn_perf", 0.042814, -0.3]
    #     test_obj=LayerTest,
    # )._commit_locate(commits)
    # print("the pr with problem is {}".format(final_commit))
    # f = open("final_commit.txt", "w")
    # f.writelines("the final commit is:{}".format(final_commit))
    # f.close()
    bs = BinarySearch(
        good_commit="651e66ba06f3ae26c3cf649f83a9a54b486ce75d",
        bad_commit="5ad596c983e6f6626a6d879b17834d15664946bc",
        layerfile="layercase/sublayer1000/Det_cases/gfl_gfl_r101vd_fpn_mstrain_2x_coco/SIR_145.py",
        testing="yaml/dy^dy2stcinn_train_inputspec.yml",
        perf_decay=None,  # ["dy2st_eval_cinn_perf", 0.042814, -0.3]
        test_obj=LayerTest,
    )
    final_commit, commit_list, commit_list_origin = bs._run()
    print("test end")
    print("final_commit:{}".format(final_commit))
    print("commit_list:{}".format(commit_list))
    print("commit_list_origin:{}".format(commit_list_origin))
