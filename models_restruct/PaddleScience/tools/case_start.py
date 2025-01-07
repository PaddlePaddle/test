# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import sys
import json
import shutil
import logging
import tarfile
import argparse
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleScience_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.getenv("reponame")
        self.mode = os.getenv("mode")
        self.case_step = os.getenv("case_step")
        self.case_name = os.getenv("case_name")
        self.qa_yaml_name = os.getenv("qa_yaml_name")

    def build_prepare(self):
        """
        执行准备过程
        """
        if "dy2st" in self.case_name:
            # logger.info("dy2st_convergence tag is: {}".format(self.case_name.split("train_")[-1]))

            # os.environ["FLAGS_prim_all"] = "false"
            # logger.info("set org FLAGS_prim_all as {}".format(os.getenv("FLAGS_prim_all")))
            os.environ["ENABLE_FALL_BACK"] = "False"
            os.environ["FLAGS_use_cinn"] = "0"
            os.environ["FLAGS_prim_all"] = "True"
            os.environ["FLAGS_prim_vjp_skip_default_ops"] = "False"
            os.environ["FLAGS_enable_pir_api"] = "True"
            logger.info("set org FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
            if "cinn" in self.case_name:
                os.environ["FLAGS_use_cinn"] = "1"
                os.environ["ENABLE_FALL_BACK"] = "False"
                logger.info("set org FLAGS_use_cinn as {}".format(os.getenv("FLAGS_use_cinn")))
        elif "deepcfd" in self.case_name:
            os.environ["FLAGS_cudnn_deterministic"] = "True"
        elif "amgnet" in self.case_name:
            py_v = os.system("python3 --version")
            print(py_v)
            os.system("python3.10 -m pip install -U pgl")
        else:
            return 0


def run():
    """
    执行入口
    """
    model = PaddleScience_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleClas_Case_Start(args)
    # model.build_prepare()
    run()
