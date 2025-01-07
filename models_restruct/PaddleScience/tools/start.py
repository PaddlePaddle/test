"""
start before model running
"""
import os
import sys
import json
import shutil
import urllib
import logging
import wget
import subprocess

logger = logging.getLogger("ce")


class PaddleScience_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        init
        """
        self.qa_yaml_name = os.environ["qa_yaml_name"]
        self.rd_yaml_path = os.environ["rd_yaml_path"]
        logger.info("###self.qa_yaml_name: {}".format(self.qa_yaml_name))
        self.reponame = os.environ["reponame"]
        self.system = os.environ["system"]
        self.step = os.environ["step"]
        logger.info("###self.step: {}".format(self.step))
        self.paddle_whl = os.environ["paddle_whl"]
        self.mode = os.environ["mode"]  # function or precision
        self.REPO_PATH = os.path.join(os.getcwd(), self.reponame)
        self.env_dict = {}
        self.model = self.qa_yaml_name.split("^")[-1]
        logger.info("###self.model_name: {}".format(self.model))
        self.env_dict["model"] = self.model
        os.environ["model"] = self.model

    def prepare_gpu_env(self):
        """
        根据操作系统获取用gpu还是cpu
        """
        if "cpu" in self.system or "mac" in self.system:
            self.env_dict["set_cuda_flag"] = "cpu"  # 根据操作系统判断
        else:
            self.env_dict["set_cuda_flag"] = "gpu"  # 根据操作系统判断
        return 0

    def add_paddlescience_to_pythonpath(self):
        """
        paddlescience 打包路径添加到python的路径中
        """
        cwd = os.getcwd()
        paddle_path = os.path.join(cwd, "PaddleScience")
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        new_pythonpath = f"{paddle_path}:{old_pythonpath}"
        os.environ["PYTHONPATH"] = new_pythonpath
        return 0

    def download_datasets(self):
        """
        download dataset
        """
        url = "https://paddle-qa.bj.bcebos.com/PaddleScience/datasets/datasets.tar.gz"
        file_name = "datasets.tar.gz"
        if not os.path.exists(file_name):
            urllib.request.urlretrieve(url, file_name)
            os.system("tar -zxvf " + file_name + " -C PaddleScience/")
            logger.info("download datasets done!!!!")
        try:
            subprocess.check_call(["python", "-m", "pip", "install", "scikit-image"])
            logger.info("Install scikit-image success!")
        except subprocess.CalledProcessError as e:
            logger.error("Install scikit-image failed. Error code: {}. Output: {}".format(e.returncode, e.output))

        try:
            subprocess.check_call(["python", "-m", "pip", "install", "pgl"])
            logger.info("Install pgl success!")
        except subprocess.CalledProcessError as e:
            logger.error("Install pgl failed. Error code: {}. Output: {}".format(e.returncode, e.output))

        try:
            subprocess.check_call(["python", "-m", "pip", "install", "hdf5storage"])
            logger.info("Install hdf5storage success!")
        except subprocess.CalledProcessError as e:
            logger.error("Install hdf5storage failed. Error code: {}. Output: {}".format(e.returncode, e.output))
        try:
            subprocess.check_call(["python", "-m", "pip", "install", "numpy==1.23.1"])
            logger.info("Install numpy success!")
        except subprocess.CalledProcessError as e:
            logger.error("Install numpy failed. Error code: {}. Output: {}".format(e.returncode, e.output))
        try:
            subprocess.check_call(["python", "-m", "pip", "install", "h5py==3.10"])
            logger.info("Install h5py success!")
        except subprocess.CalledProcessError as e:
            logger.error("Install h5py failed. Error code: {}. Output: {}".format(e.returncode, e.output))

    def build_prepare(self):
        """
        build prepare
        """
        ret = 0
        ret = self.prepare_gpu_env()
        if ret:
            logger.info("build prepare_gpu_env failed")
            return ret
        os.environ[self.reponame] = json.dumps(self.env_dict)
        return ret

    def install_env(self):
        """
        install env for  paddlescience docker
        """
        mode = os.getenv("mode")
        system = os.environ["system"]
        if mode == "function" and "Windows" not in system:
            os.system("apt update")
            os.system("apt-get install -y curl")
            os.system("curl --version")


def run():
    """
    执行入口
    """
    model = PaddleScience_Start()
    model.build_prepare()
    model.add_paddlescience_to_pythonpath()
    model.download_datasets()
    model.install_env()
    return 0


if __name__ == "__main__":
    run()
