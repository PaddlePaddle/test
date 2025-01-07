#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
测试执行器
"""
import os
import shutil
import subprocess
from subprocess import TimeoutExpired
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import platform
from datetime import datetime
import pandas as pd
import layertest
from db.layer_db import LayerBenchmarkDB
from strategy.compare import perf_compare_dict, perf_compare_kernel_dict
from pltools.case_select import CaseSelect
from pltools.logger import Logger
from pltools.yaml_loader import YamlLoader
from pltools.json_loader import JSONLoader
from pltools.res_save import xlsx_save, download_sth, create_tar_gz, extract_tar_gz, load_pickle, save_txt
from pltools.nv_tool import get_nv_memory
from pltools.upload_bos import UploadBos
from pltools.statistics import split_list, sublayer_perf_gsb_gen, kernel_perf_gsb_gen, sublayer_perf_ratio_gen
from pltools.alarm import Alarm


class Run(object):
    """
    最终执行接口
    """

    def __init__(self):
        """
        init
        """
        # 获取所有layer.yml文件路径
        self.layer_type = os.environ.get("CASE_TYPE")
        self.layer_dir = [os.path.join(self.layer_type, item) for item in os.environ.get("CASE_DIR").split(",")]

        # 获取需要忽略的case
        self.ignore_list = YamlLoader(yml=os.path.join("yaml", "ignore_case.yml")).yml.get(os.environ.get("TESTING"))

        # 获取测试集
        # self.py_list = CaseSelect(self.layer_dir, self.ignore_list).get_py_list(base_path=self.layer_dir)
        py_list = []
        for layer_dir in self.layer_dir:
            py_list = py_list + CaseSelect(layer_dir, self.ignore_list).get_py_list(base_path=layer_dir)

        # 测试集去重
        self.py_list = []
        for item in py_list:
            if not item in self.py_list:
                self.py_list.append(item)

        self.testing = os.environ.get("TESTING")
        self.py_cmd = os.environ.get("python_ver")
        self.report_dir = os.path.join(os.getcwd(), "report")

        self.logger = Logger("PaddleLTRun")
        self.AGILE_PIPELINE_BUILD_ID = os.environ.get("AGILE_PIPELINE_BUILD_ID", 0)
        self.now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.storage = "apibm_config.yml"

        if os.environ.get("FRAMEWORK") == "paddle":
            import paddle

            self.logger.get_log().info(f"Paddle框架commit: {paddle.__git_commit__}, 版本: {paddle.__version__}")
            os.environ["paddle_commit"] = paddle.__git_commit__
            if os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleOCR":
                os.system(
                    "wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleOCR/PaddleOCR.tar.gz --no-proxy "
                    "&& tar -xzf PaddleOCR.tar.gz && rm -rf PaddleOCR.tar.gz "
                    "&& cd PaddleOCR && git rev-parse HEAD && git branch "
                    f"&& {self.py_cmd} -m pip install -r requirements.txt && {self.py_cmd} setup.py install "
                    # f"&& {self.py_cmd} -m pip install paddlenlp "
                )
            elif os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleNLP":
                os.system(
                    "wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleNLP/PaddleNLP-develop.tar.gz --no-proxy "
                    "&& tar -xzf PaddleNLP-develop.tar.gz && rm -rf PaddleNLP-develop.tar.gz "
                    "&& cd PaddleNLP-develop && git rev-parse HEAD && git branch "
                    f"&& {self.py_cmd} -m pip install -r requirements.txt && {self.py_cmd} setup.py install "
                    f"&& {self.py_cmd} -m pip install yacs && {self.py_cmd} -m pip install sacremoses "
                    f"&& {self.py_cmd} -m pip install emoji && {self.py_cmd} -m pip install ftfy "
                    f"&& {self.py_cmd} -m pip install unidecode "
                )
        elif os.environ.get("FRAMEWORK") == "torch":
            self.logger.get_log().info("开始安转torch release版本")
            os.system(
                f"{self.py_cmd} -m pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu118"
            )
            import torch

            self.logger.get_log().info(f"Torch框架版本: {torch.__version__}")

        # 下载ground truth用于跨硬件测试
        plt_gt_download_url = os.environ.get("PLT_GT_DOWNLOAD_URL")
        if not plt_gt_download_url == "None" and os.environ.get("TESTING_MODE") == "precision":
            self.logger.get_log().info(f"下载plt_gt的url为: {plt_gt_download_url}")
            plt_gt_device = plt_gt_download_url.split("/")[-1]
            # if not os.path.exists(os.path.join("plt_gt_baseline", plt_gt_device)):
            #     os.makedirs(os.path.join("plt_gt_baseline", plt_gt_device))
            for testing in YamlLoader(yml=self.testing).get_junior_name("testings"):
                if not os.path.exists(os.path.join("plt_gt_baseline", plt_gt_device, testing)):
                    os.makedirs(os.path.join("plt_gt_baseline", plt_gt_device, testing))
                for py_file in self.py_list:
                    case_name = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
                    self.logger.get_log().info(f"开始下载plt_gt: {case_name}")
                    gt_url = f"{plt_gt_download_url}/{testing}/{case_name}.tensor"
                    download_sth(
                        gt_url=gt_url,
                        output_path=os.path.join("plt_gt_baseline", plt_gt_device, testing, f"{case_name}.tensor"),
                    )

    def _exit_code_txt(self, error_count, error_list):
        """"""
        core_dumps_list = self._core_dumps_case_count(report_path=self.report_dir)
        if error_count != 0 or core_dumps_list:
            self.logger.get_log().warning("测试失败, 下面进行bug分类统计: ")
            self.logger.get_log().warning(f"报错为core dumps的子图有: {core_dumps_list}")
            self.logger.get_log().info(f"测试子图总数为: {len(self.py_list)}")
            self.logger.get_log().warning(f"报错子图总数为: {len(error_list)}")
            self.logger.get_log().warning(f"报错为core dumps的子图数量为: {len(core_dumps_list)}")
            self.logger.get_log().warning(f"报错不为core dumps的异常子图数量为: {len(error_list)-len(core_dumps_list)}")
            self.logger.get_log().warning("请注意, 由于程序崩溃, core dumps的子图不会出现在allure报告中")
            self.logger.get_log().info(
                "『测试子图总数』 = 『下方回调'total'个数(allure报告中case总数)』 + 『core dump崩溃子图数』, 请核对case数目以保证测试无遗漏"
            )
            os.system("echo 7 > exit_code.txt")
        else:
            self.logger.get_log().info(f"测试子图总数为: {len(self.py_list)}")
            self.logger.get_log().info("测试通过, 无报错子图-。-")
            os.system("echo 0 > exit_code.txt")

    def _db_interact(self, sublayer_dict, error_list):
        """Database interaction"""
        # 数据库交互
        if os.environ.get("PLT_BM_DB") == "insert":  # 存入数据, 作为基线或对比
            layer_db = LayerBenchmarkDB(storage=self.storage)
            if os.environ.get("PLT_BM_MODE") == "baseline":
                layer_db.baseline_insert(data_dict=sublayer_dict, error_list=error_list)

                return {}, "none"
            elif os.environ.get("PLT_BM_MODE") == "latest_as_baseline":
                baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()

                # 先获取baseline, 再录入新的baseline
                layer_db.baseline_insert(data_dict=sublayer_dict, error_list=error_list)

                return baseline_dict, baseline_layer_type  # 实际返回的是前一次baseline
            elif os.environ.get("PLT_BM_MODE") == "latest":
                baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()
                layer_db.latest_insert(data_dict=sublayer_dict, error_list=error_list)

                return baseline_dict, baseline_layer_type
            else:
                raise Exception(
                    "unknown benchmark mode, PaddleLT benchmark only support "
                    "baseline mode, latest_as_baseline mode or latest mode"
                )
        elif os.environ.get("PLT_BM_DB") == "select":  # 不存数据, 仅对比并生成表格
            layer_db = LayerBenchmarkDB(storage=self.storage)
            baseline_dict, baseline_layer_type = layer_db.get_baseline_dict()

            return baseline_dict, baseline_layer_type
        elif os.environ.get("PLT_BM_DB") == "non-db":  # 不加载数据库，仅生成表格

            return "none", "none"
        else:
            Exception("unknown benchmark datebase mode, only support insert, select or non-db")

    def _gt_upload(self):
        """精度groundtruth上传"""
        upload_url = os.environ.get("PLT_GT_UPLOAD_URL")
        if not upload_url == "None":
            _upload = UploadBos()
            self.logger.get_log().info(f"上传plt_gt的路径为: {os.environ.get('PLT_GT_UPLOAD_URL')}")
            for device in os.listdir("plt_gt"):
                device_path = os.path.join("plt_gt", device)
                for testing in os.listdir(device_path):
                    testing_path = os.path.join(device_path, testing)
                    for tensor in os.listdir(testing_path):
                        _upload.upload_to_bos(
                            bos_path=os.path.join(upload_url, device, testing),
                            file_path=os.path.join(testing_path, tensor),
                        )

    def _perf_upload(self):
        """性能表格/图表/原始数据上传"""
        bos_path = "PaddleLT/PaddleLTBenchmark/{}/build_{}".format(
            os.environ.get("PLT_BM_DB"), self.AGILE_PIPELINE_BUILD_ID
        )
        excel_file = os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx"
        if os.path.exists(excel_file):
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path=excel_file)
            self.logger.get_log().info("表格下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, excel_file))

        if os.path.exists("gsb_dict.txt"):
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path="gsb_dict.txt")
            self.logger.get_log().info(
                "GSB.txt下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, "gsb_dict.txt")
            )

        if os.environ.get("PLT_BM_PLOT") == "True":
            os.system("tar -czf plot.tar *.png")
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path="plot.tar")
            self.logger.get_log().info("plot下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, "plot.tar"))
            os.system("tar -czf pickle.tar *.pickle")
            UploadBos().upload_to_bos(bos_path="paddle-qa/{}".format(bos_path), file_path="pickle.tar")
            self.logger.get_log().info(
                "pickle下载链接: https://paddle-qa.bj.bcebos.com/{}/{}".format(bos_path, "pickle.tar")
            )

        if os.environ.get("PLT_BM_EMAIL") == "True":
            desc = os.environ.get("TESTING").split("/")[-1].split(".")[0]
            alarm = Alarm(storage=self.storage)
            alarm.email_send(
                alarm.receiver,
                f"Paddle {self.layer_type}子图性能数据{desc}",
                f"表格下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/{excel_file}\n"
                f"\n"
                f"GSB.txt下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/gsb_dict.txt\n"
                f"\n"
                # f"plot下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/plot.tar\n"
                # f"pickle下载链接: https://paddle-qa.bj.bcebos.com/{bos_path}/pickle.tar",
            )

    def _single_pytest_run(self, py_file, testing, device_place_id=0):
        """run one test"""
        title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
        self.logger.get_log().info(f"开始测试子图 {title}, 准备执行pytest命令~~")

        if os.environ.get("PLT_PYTEST_TIMEOUT") == "None":
            if self.layer_type == "layerE2Ecase":
                exit_code = os.system(f"{self.py_cmd} -m pytest {py_file} --alluredir={self.report_dir}")
            else:
                exit_code = os.system(
                    "cp -r PaddleLT.py {}.py && "
                    "{} -m pytest {}.py --title={} --layerfile={} --testing={} "
                    "--device_place_id={} --alluredir={}".format(
                        title, self.py_cmd, title, title, py_file, testing, device_place_id, self.report_dir
                    )
                )
        else:
            timeout = os.environ.get("PLT_PYTEST_TIMEOUT")
            if self.layer_type == "layerE2Ecase":
                cmd = f"{self.py_cmd} -m pytest {py_file} --alluredir={self.report_dir} --timeout={timeout}"
            else:
                cmd = (
                    "cp -r PaddleLT.py {}.py && "
                    "{} -m pytest {}.py --title={} --layerfile={} --testing={} "
                    "--device_place_id={} --alluredir={} --timeout={}"
                ).format(title, self.py_cmd, title, title, py_file, testing, device_place_id, self.report_dir, timeout)

            # 使用subprocess执行命令并设置超时
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = proc.communicate(timeout=float(timeout))
                exit_code = proc.returncode
                self.logger.get_log().warning(f"{py_file} Command failed with return code {exit_code}")
                # 如果进程正常结束，stdout 和 stderr 将包含输出
                if stdout:
                    self.logger.get_log().info(stdout.decode())  # 注意：输出是字节串，需要解码为字符串
                if stderr:
                    self.logger.get_log().warning(stderr.decode())  # 注意：错误输出也是字节串，需要解码为字符串
                if exit_code != 0:
                    self.logger.get_log().warning(f"{py_file} Command failed with return code {exit_code}")
            except subprocess.TimeoutExpired:
                self.logger.get_log().warning(f"{py_file} Command timed out after {timeout} seconds")
                proc.terminate()  # 发送 SIGTERM 信号到进程
                exit_code = -1

        self.logger.get_log().info(f"完成测试子图 {title}, 完成执行pytest命令~~")
        if exit_code != 0:
            return py_file, exit_code
        return None, None

    def _multithread_test_run(self, py_list):
        """multithread run some test"""
        error_list = []
        error_count = 0

        with ThreadPoolExecutor(max_workers=int(os.environ.get("MULTI_WORKER", 13))) as executor:
            # 提交任务给线程池
            futures = [executor.submit(self._single_pytest_run, py_file, self.testing) for py_file in py_list]

            # 等待任务完成，并收集返回值
            for future in futures:
                _py_file, _exit_code = future.result()
                if _exit_code is not None:
                    error_list.append(_py_file)
                    error_count += 1

        if os.environ.get("MULTI_DOUBLE_CHECK") == "False":
            if not os.environ.get("PLT_GT_UPLOAD_URL") == "None":
                self._gt_upload()
            self._exit_code_txt(error_count=error_count, error_list=error_list)
        else:
            self.logger.get_log().info("对于多线程失败case, 进入double check环节: ")
            self._test_run(py_list=error_list)

    def _multi_gpu_multithread_test_run(self, py_list):
        """multithread run some test"""
        ######################################################
        def _queue_run(py_list, device_place_id, result_queue):
            # def _queue_run(py_list, py_dict, result_queue):
            """
            multi run main
            """
            error_list = []
            error_count = 0

            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_place_id)
            with ThreadPoolExecutor(max_workers=int(os.environ.get("MULTI_WORKER", 13))) as executor:
                # 提交任务给线程池
                # futures = [
                #     executor.submit(self._single_pytest_run, py_file, self.testing, py_dict[py_file])
                #     for py_file in py_list
                # ]

                futures = [executor.submit(self._single_pytest_run, py_file, self.testing, 0) for py_file in py_list]

                # 等待任务完成，并收集返回值
                for future in futures:
                    _py_file, _exit_code = future.result()
                    if _exit_code is not None:
                        error_list.append(_py_file)
                        error_count += 1

            result_queue.put((error_list, error_count))

        ######################################################

        device_list = [int(x) for x in os.environ.get("PLT_DEVICE_ID").split(",")]
        if len(device_list) < 2:
            raise Exception("single gpu cannot use _multi_gpu_multithread_test_run strategy")

        # py_dict = {item: i % len(device_list) for i, item in enumerate(py_list)}

        multiprocess_cases = split_list(lst=self.py_list, n=len(device_list))
        processes = []
        result_queue = multiprocessing.Queue()

        for i, cases_list in enumerate(multiprocess_cases):
            self.logger.get_log().info(f"multiprocess_cases中i: {i}")
            self.logger.get_log().info(f"multiprocess_cases中device_list: {device_list}")
            self.logger.get_log().info(f"multiprocess_cases中device_list[i]: {device_list[i]}")
            process = multiprocessing.Process(target=_queue_run, args=(cases_list, device_list[i], result_queue))
            # process = multiprocessing.Process(target=_queue_run, args=(cases_list, py_dict, result_queue))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        error_list = []
        error_count = 0
        while not result_queue.empty():
            single_error_list, single_error_count = result_queue.get()
            error_list.extend(single_error_list)
            error_count += single_error_count

        if os.environ.get("MULTI_DOUBLE_CHECK") == "False":
            if not os.environ.get("PLT_GT_UPLOAD_URL") == "None":
                self._gt_upload()
            self._exit_code_txt(error_count=error_count, error_list=error_list)
        else:
            self.logger.get_log().info("对于多线程失败case, 进入double check环节: ")
            self._test_run(py_list=error_list)
        ######################################################

        # error_list = []
        # error_count = 0

        # device_list = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]
        # if len(device_list) < 2:
        #     raise Exception("single gpu cannot use _multi_gpu_multithread_test_run strategy")

        # py_dict = {item: i % len(device_list) for i, item in enumerate(py_list)}
        # with ThreadPoolExecutor(max_workers=int(os.environ.get("MULTI_WORKER", 13))) as executor:
        #     # 提交任务给线程池
        #     futures = [
        #         executor.submit(self._single_pytest_run, py_file, self.testing, device_place_id)
        #         for py_file, device_place_id in py_dict.items()
        #     ]

        #     # 等待任务完成，并收集返回值
        #     for future in futures:
        #         _py_file, _exit_code = future.result()
        #         if _exit_code is not None:
        #             error_list.append(_py_file)
        #             error_count += 1

        # if os.environ.get("MULTI_DOUBLE_CHECK") == "False":
        #     if not os.environ.get("PLT_GT_UPLOAD_URL") == "None":
        #         self._gt_upload()
        #     self._exit_code_txt(error_count=error_count, error_list=error_list)
        # else:
        #     self.logger.get_log().info("对于多线程失败case, 进入double check环节: ")
        #     self._test_run(py_list=error_list)

    def _test_run(self, py_list):
        """run some test"""
        sublayer_dict = {}
        error_list = []
        error_count = 0
        for py_file in py_list:
            if os.environ.get("PLT_GET_NV_MEMORY") == "True":
                self.logger.get_log().info(get_nv_memory(int(os.environ.get("PLT_DEVICE_ID"))))
            _py_file, _exit_code = self._single_pytest_run(py_file=py_file, testing=self.testing)
            if _exit_code is not None:
                error_list.append(_py_file)
                error_count += 1
                prec_dict = {self.testing: "fail"}
            else:
                prec_dict = {self.testing: "pass"}

            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            sublayer_dict[title] = prec_dict

        if not os.environ.get("PLT_GT_UPLOAD_URL") == "None":
            self._gt_upload()

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        if os.environ.get("PLT_BM_DB") != "non-db":
            baseline_dict, baseline_layer_type = self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)

    def _multiprocess_perf_test_run(self):
        """
        multiprocess run some test
        需要处理cpu绑核以及gpu并行
        """

        def _queue_run(py_list, device_id, result_queue):
            """
            multi run main
            """
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PLT_DEVICE_ID") + str(device_id)
            sublayer_dict = {}
            error_count = 0
            error_list = []
            for py_file in py_list:
                title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
                single_test = layertest.LayerTest(title=title, layerfile=py_file, testing=self.testing)
                perf_dict, exit_code = single_test._perf_case_run()

                # 报错的子图+engine将不会收录进sublayer_dict
                if exit_code != 0:
                    error_list.append(py_file)
                    error_count += 1
                    continue

                sublayer_dict[title] = perf_dict

            # error_dict = self._run_main(all_cases=all_cases, loops=loops, base_times=base_times)

            result_queue.put(sublayer_dict, error_list, error_count)

        multiprocess_cases = split_list(lst=self.py_list, n=int(os.environ.get("MULTI_WORKER")))
        processes = []
        result_queue = multiprocessing.Queue()

        for i, cases_list in enumerate(multiprocess_cases):
            process = multiprocessing.Process(target=_queue_run, args=(cases_list, i, result_queue))
            process.start()
            # os.sched_setaffinity(process.pid, {self.core_index + i})
            processes.append(process)

        for process in processes:
            process.join()

        sublayer_dict = {}
        error_list = []
        error_count = 0
        compare_list = YamlLoader(yml=self.testing).yml.get("compare")
        while not result_queue.empty():
            single_sublayer_dict, single_error_list, single_error_count = result_queue.get()
            sublayer_dict.update(single_sublayer_dict)
            error_list.extend(single_error_list)
            error_count += single_error_count

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        baseline_dict, baseline_layer_type = self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)
        self._perf_report_gen(
            compare_list=compare_list,
            baseline_dict=baseline_dict,
            sublayer_dict=sublayer_dict,
            error_list=error_list,
            baseline_layer_type=baseline_layer_type,
            latest_layer_type=self.layer_type,
        )

        self._perf_upload()
        self._pts_callback(error_count)

    def _perf_test_run(self):
        """run some test"""
        sublayer_dict = {}
        error_count = 0
        error_list = []
        compare_list = YamlLoader(yml=self.testing).yml.get("compare")
        for py_file in self.py_list:
            if os.environ.get("PLT_BM_ERROR_CHECK") == "True":  # 先跑功能看是否能通过
                _py_file, _exit_code = self._single_pytest_run(
                    py_file=py_file, testing="yaml/pre-dy2stcinn_train_bm.yml"
                )
                if _exit_code is not None:
                    error_list.append(_py_file)
                    error_count += 1
                    continue

            title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
            single_test = layertest.LayerTest(title=title, layerfile=py_file, testing=self.testing)
            perf_dict, exit_code = single_test._perf_case_run()

            # 报错的子图+engine将不会收录进sublayer_dict
            if exit_code != 0:
                error_list.append(py_file)
                error_count += 1
                continue

            sublayer_dict[title] = perf_dict

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        baseline_dict, baseline_layer_type = self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)
        self._perf_report_gen(
            compare_list=compare_list,
            baseline_dict=baseline_dict,
            sublayer_dict=sublayer_dict,
            error_list=error_list,
            baseline_layer_type=baseline_layer_type,
            latest_layer_type=self.layer_type,
        )

        self._perf_upload()
        self._pts_callback(error_count)

    def _perf_unit_test_run(self):
        """run some test"""
        if os.path.exists("./nv_report"):
            shutil.rmtree("./nv_report")
            self.logger.get_log().info("已删除./nv_report路径以及其内容")
        os.makedirs(name="./nv_report")

        sublayer_dict = {}
        error_count = 0
        error_list = []
        testings_list = YamlLoader(yml=self.testing).get_junior_name("testings")
        compare_list = YamlLoader(yml=self.testing).yml.get("compare")
        for py_file in self.py_list:
            perf_dict = {}
            for plt_exc in testings_list:
                title = py_file.replace(".py", "").replace("/", "^").replace(".", "^")
                if os.environ.get("PLT_PYTEST_TIMEOUT") == "None":
                    if os.environ.get("PLT_PERF_CONTENT") == "kernel":
                        exit_code = os.system(
                            f"nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas "
                            f"-o nv_report/{title}-{plt_exc} {self.py_cmd} "
                            f"layertest.py --layerfile {py_file} --testing {self.testing} --plt_exc {plt_exc}"
                        )
                    else:
                        exit_code = os.system(
                            f"layertest.py --layerfile {py_file} --testing {self.testing} --plt_exc {plt_exc}"
                        )
                else:
                    timeout = os.environ.get("PLT_PYTEST_TIMEOUT")
                    if os.environ.get("PLT_PERF_CONTENT") == "kernel":
                        cmd = (
                            f"nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas "
                            f"-o nv_report/{title}-{plt_exc} {self.py_cmd} "
                            f"layertest.py --layerfile {py_file} --testing {self.testing} --plt_exc {plt_exc}"
                        )
                    else:
                        cmd = f"layertest.py --layerfile {py_file} --testing {self.testing} --plt_exc {plt_exc}"
                    # 使用subprocess执行命令并设置超时
                    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        stdout, stderr = proc.communicate(timeout=float(timeout))
                        exit_code = proc.returncode
                        self.logger.get_log().warning(f"{py_file} Command failed with return code {exit_code}")
                        # 如果进程正常结束，stdout 和 stderr 将包含输出
                        if stdout:
                            self.logger.get_log().info(stdout.decode())  # 注意：输出是字节串，需要解码为字符串
                        if stderr:
                            self.logger.get_log().warning(stderr.decode())  # 注意：错误输出也是字节串，需要解码为字符串
                        if exit_code != 0:
                            self.logger.get_log().warning(f"{py_file} Command failed with return code {exit_code}")
                    except subprocess.TimeoutExpired:
                        self.logger.get_log().warning(f"{py_file} Command timed out after {timeout} seconds")
                        proc.terminate()  # 发送 SIGTERM 信号到进程
                        exit_code = -1

                try:  # 兼容core dump不产出loaded_data的情况
                    loaded_data = load_pickle(
                        filename=os.path.join("perf_unit_result", title + "-" + plt_exc + ".pickle")
                    )
                    # 报错的子图+engine将不会收录进sublayer_dict
                    if exit_code != 0 or loaded_data[-1] > 0:
                        if py_file not in error_list:
                            error_list.append(py_file)
                            error_count += 1
                        continue
                except Exception:  # 兼容core dump不产出loaded_data的情况
                    if py_file not in error_list:
                        error_list.append(py_file)
                        error_count += 1
                    continue

                if os.environ.get("PLT_PERF_CONTENT") == "kernel":
                    os.system(
                        f"nsys stats --report cuda_gpu_kern_sum --format csv "
                        f"--output . "
                        f"nv_report/{title}-{plt_exc}.sqlite"
                    )
                    df = pd.read_csv(os.path.join("nv_report", f"{title}-{plt_exc}_cuda_gpu_kern_sum.csv"))
                    kernel_time = (df["Instances"] * df["Avg (ns)"]).sum()
                    kernel_count = df["Instances"].sum()

                    perf_dict[plt_exc + "-" + "kernel_time"] = kernel_time
                    perf_dict[plt_exc + "-" + "kernel_count"] = int(kernel_count)  # int64无法json化, 所以先转为通用int
                    self.logger.get_log().info("kernel time and count: ")
                    self.logger.get_log().info(f"kernel time is {kernel_time}")
                    self.logger.get_log().info(f"kernel count is {kernel_count}")
                else:
                    perf_dict[plt_exc] = loaded_data[0][plt_exc]

            sublayer_dict[title] = perf_dict

        self._exit_code_txt(error_count=error_count, error_list=error_list)

        baseline_dict, baseline_layer_type = self._db_interact(sublayer_dict=sublayer_dict, error_list=error_list)
        self._perf_report_gen(
            compare_list=compare_list,
            baseline_dict=baseline_dict,
            sublayer_dict=sublayer_dict,
            error_list=error_list,
            baseline_layer_type=baseline_layer_type,
            latest_layer_type=self.layer_type,
        )

        self._perf_upload()
        self._pts_callback(error_count)

    def _perf_report_gen(
        self, compare_list, baseline_dict, sublayer_dict, error_list, baseline_layer_type, latest_layer_type
    ):
        """
        精度对比策略
        """
        if baseline_layer_type != "none" and (
            os.environ.get("PLT_BM_MODE") == "latest_as_baseline" or os.environ.get("PLT_BM_MODE") == "latest"
        ):
            if os.environ.get("PLT_PERF_CONTENT") == "kernel":
                compare_dict = perf_compare_kernel_dict(
                    compare_list=compare_list,
                    baseline_dict=baseline_dict,
                    data_dict=sublayer_dict,
                    error_list=error_list,
                    baseline_layer_type=baseline_layer_type,
                    latest_layer_type=self.layer_type,
                )
                gsb_dict = kernel_perf_gsb_gen(compare_dict=compare_dict, compare_list=compare_list)
            else:
                compare_dict = perf_compare_dict(
                    compare_list=compare_list,
                    baseline_dict=baseline_dict,
                    data_dict=sublayer_dict,
                    error_list=error_list,
                    baseline_layer_type=baseline_layer_type,
                    latest_layer_type=self.layer_type,
                )
                gsb_dict = sublayer_perf_gsb_gen(compare_dict=compare_dict, compare_list=compare_list)
                ratio_dict = sublayer_perf_ratio_gen(compare_dict=compare_dict, compare_list=compare_list)
                for key, value in gsb_dict.items():
                    gsb_dict[key] = {**gsb_dict[key], **ratio_dict[key]}
            save_txt(data=gsb_dict, filename="gsb_dict")
            xlsx_save(
                sublayer_dict=compare_dict,
                excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
            )
        else:
            xlsx_save(
                sublayer_dict=sublayer_dict,
                excel_file=os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx",
            )

    def _core_dumps_case_count(self, report_path):
        """
        统计allure报告中无法展示的, core dumps程序崩溃的case
        """
        # 性能测试无allure report, 直接返回空
        if not os.path.exists(report_path):
            return []

        allure_case_list = []
        # sub_case_list = [] # 用于单个py文件中多个case统计
        for json_file in os.listdir(report_path):
            if json_file.endswith("-result.json"):
                # layerE2Ecase中allure报告需要抓取的关键字, 与其他子图不一样
                if self.layer_type == "layerE2Ecase":
                    if "fullName" in JSONLoader(os.path.join(report_path, json_file)).json_dict():
                        case_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["fullName"]
                        layer_name = case_name[: case_name.rfind(".")]
                        allure_case_list.append(layer_name.replace(".", "/") + ".py")
                    elif "labels" in JSONLoader(os.path.join(report_path, json_file)).json_dict():
                        layer_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["labels"][-1]["value"]
                        allure_case_list.append(layer_name.replace(".", "/") + ".py")

                    # # layer_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["labels"][-1]["value"]
                    # # allure_case_list.append(layer_name.replace(".", "/") + ".py")
                    # case_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["fullName"]
                    # layer_name = case_name[:case_name.rfind('.')]
                    # allure_case_list.append(layer_name.replace(".", "/") + ".py")
                    # # sub_case_list.append(case_name)
                else:
                    layer_name = JSONLoader(os.path.join(report_path, json_file)).json_dict()["name"]
                    allure_case_list.append(layer_name.replace("^", "/") + ".py")

        all_case_list = []
        # 将./layercase/sublayer1000/Det_cases/gfl_gflv2_r50_fpn_1x_coco/SIR_173.py
        # 转换为layercase^sublayer1000^Det_cases^gfl_gflv2_r50_fpn_1x_coco^SIR_173
        for py_file in self.py_list:
            all_case_list.append(py_file)

        # 如果allure报告中不包含某个case, 说明这个case出现了core dumps
        core_dumps_list = []
        for case in all_case_list:
            if case not in allure_case_list:
                core_dumps_list.append(case)

        return core_dumps_list

    def _pts_callback(self, error_count):
        """
        用于性能任务回调pts. 精度任务通过start.sh最后的命令回调
        """
        pts_id = os.environ.get("pts_id")
        bos_path = "PaddleLT/PaddleLTBenchmark/{}/build_{}".format(
            os.environ.get("PLT_BM_DB"), self.AGILE_PIPELINE_BUILD_ID
        )
        excel_file = os.environ.get("TESTING").replace("yaml/", "").replace(".yml", "") + ".xlsx"
        report_url = f"https://paddle-qa.bj.bcebos.com/{bos_path}/{excel_file}"

        success_count = len(self.py_list) - error_count
        if error_count > 0:
            status = "失败"
            result = f"success:{success_count} fail:{error_count}"
        else:
            status = "成功"
            result = f"success:{success_count} fail:{error_count}"

        os.system(f"bash ./callback.sh '{pts_id}' '{status}' '{result}' '{report_url}'")


if __name__ == "__main__":
    tes = Run()
    if os.environ.get("TESTING_MODE") == "precision":
        if os.environ.get("MULTI_WORKER") == "0":
            tes._test_run(py_list=tes.py_list)
        else:
            tes._multithread_test_run(py_list=tes.py_list)
    elif os.environ.get("TESTING_MODE") == "performance":
        if os.environ.get("PLT_PERF_MODE") == "unit-python":
            tes._perf_unit_test_run()
        else:
            if os.environ.get("MULTI_WORKER") == "0":
                tes._perf_test_run()
            else:
                tes._multiprocess_perf_test_run()
    elif os.environ.get("TESTING_MODE") == "precision_multi_gpu":
        tes._multi_gpu_multithread_test_run(py_list=tes.py_list)
    else:
        raise Exception("unknown testing mode, PaddleLayerTest only support test precision or performance")
