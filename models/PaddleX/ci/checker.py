# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""

import os
import json
import shutil
import tarfile
import argparse
import requests
from tqdm import tqdm
from yaml import safe_load, dump
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


def parse_args():
    """Parse the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dataset", action="store_true", default=False)
    parser.add_argument("--module_name", type=str, default=False)
    parser.add_argument("--config_path", type=str, default=False)
    parser.add_argument("--dataset_url", type=str, default=False)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=False)
    parser.add_argument("--check_weights_items", type=str, default=False)

    parser.add_argument("--check_train_result_json", action="store_true", default=False)
    parser.add_argument("--check_train_config_content", action="store_true", default=False)
    parser.add_argument("--check_dataset_result", action="store_true", default=False)
    parser.add_argument("--check_split_dataset", action="store_true", default=False)
    parser.add_argument("--check_eval_result_json", action="store_true", default=False)
    all_arguments = [action.dest for action in parser._actions if action.dest]
    check_items = []
    for arg in all_arguments:
        if "check_" in arg:
            check_items.append(arg)
    args = parser.parse_args()
    return check_items, args


def download_dataset(args):
    """Download dataset"""
    with open(args.config_path, "r") as file:
        dataset_info = safe_load(file)
    dataset_dir = dataset_info["Global"]["dataset_dir"].rstrip("/")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    dataset_parent_dir = os.path.dirname(dataset_dir)
    if not os.path.exists(dataset_parent_dir):
        os.makedirs(dataset_parent_dir)
    save_path = os.path.join(dataset_parent_dir, args.dataset_url.split("/")[-1])

    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    response = session.head(args.dataset_url, allow_redirects=True)
    file_size = int(response.headers.get("content-length", 0))
    with session.get(args.dataset_url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()

        with open(save_path, "wb") as f:
            pbar = tqdm(total=file_size, unit="iB", unit_scale=True, desc=args.dataset_url.split("/")[-1])
            for data in r.iter_content(chunk_size=8192):
                f.write(data)
                pbar.update(len(data))
        pbar.close()

    with tarfile.open(save_path, "r") as tar:
        tar.extractall(path=dataset_parent_dir)
    os.remove(save_path)


class PostTrainingChecker:
    """Post training checker class"""

    def __init__(self, args):
        self.check_results = []
        self.check_flag = []

    def check_train_json_content(
        self, output_dir, module_name, check_weights_items, train_result_json, check_train_json_message
    ):
        """Check train result json content"""
        pass_flag = True
        if not os.path.exists(train_result_json):
            check_train_json_message.append(f"检查失败：{train_result_json} 不存在.")
            pass_flag = False

        try:
            with open(train_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            check_train_json_message.append(f"打开 {train_result_json} 文件失败.")
            pass_flag = False

        if not json_data.get("done_flag", False):
            check_train_json_message.append("检查失败：训练未完成")
            pass_flag = False
            err_type = json_data.get("err_type", None)
            err_msg = json_data.get("err_msg", None)
            if err_type and err_msg:
                check_train_json_message.append(f"报错类型：{err_type}")
                check_train_json_message.append(f"报错信息：{err_msg}")
            else:
                check_train_json_message.append("检查失败：未正确返回报错信息")
        else:
            if "ts" in module_name:
                inspection_item = [
                    "score",
                    "pdparams",
                ]
                last_data = json_data["models"]["best"]

                for file_key in inspection_item:
                    if file_key == "score":
                        score = last_data.get(file_key)
                        if score == "":
                            check_train_json_message.append(f"检查失败：{file_key} 不存在")
                            pass_flag = False
                    else:
                        file_path = os.path.join(output_dir, last_data.get(file_key))
                        if last_data.get(file_key) == "" or not os.path.exists(file_path):
                            check_train_json_message.append(f"检查失败：在best中，{file_key} 对应的文件 {file_path} 不存在或为空")
                            pass_flag = False
            else:
                config_path = json_data.get("config")
                visualdl_log_path = json_data.get("visualdl_log")
                label_dict_path = json_data.get("label_dict")
                if not os.path.exists(os.path.join(output_dir, config_path)):
                    check_train_json_message.append(f"检查失败：配置文件 {config_path} 不存在")
                    pass_flag = False
                if not ("text" in module_name or "table" in module_name or "formula" in module_name):
                    if not os.path.exists(os.path.join(output_dir, visualdl_log_path)):
                        check_train_json_message.append(f"检查失败：VisualDL日志文件 {visualdl_log_path} 不存在")
                        pass_flag = False

                if not os.path.exists(os.path.join(output_dir, label_dict_path)):
                    check_train_json_message.append(f"检查失败：标签映射文件 {label_dict_path} 不存在")
                    pass_flag = False

                inspection_item = check_weights_items.split(",")[1:]
                last_k = check_weights_items.split(",")[0]
                for i in range(1, int(last_k)):
                    last_key = f"last_{i}"
                    last_data = json_data["models"].get(last_key)

                    for file_key in inspection_item:
                        file_path = os.path.join(output_dir, last_data.get(file_key))
                        if last_data.get(file_key) == "" or not os.path.exists(file_path):
                            check_train_json_message.append(f"检查失败：在 {last_key} 中，{file_key} 对应的文件 {file_path} 不存在或为空")
                            pass_flag = False

                best_key = "best"
                best_data = json_data["models"].get(best_key)
                if best_data.get("score") == "":
                    check_train_json_message.append(f"检查失败：{best_key} 中，score 不存在或为空")
                    pass_flag = False
                for file_key in inspection_item:
                    file_path = os.path.join(output_dir, best_data.get(file_key))
                    if best_data.get(file_key) == "" or not os.path.exists(file_path):
                        check_train_json_message.append(f"检查失败：在 {best_key} 中，{file_key} 对应的文件 {file_path} 不存在或为空")
                        pass_flag = False
        return pass_flag, check_train_json_message

    def check_dataset_json_content(self, output_dir, module_name, dataset_result_json, check_dataset_json_message):
        """Check dataset result json content"""
        pass_flag = True
        if not os.path.exists(dataset_result_json):
            check_dataset_json_message.append(f"{dataset_result_json} 不存在.")

        try:
            with open(dataset_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            check_dataset_json_message.append(f"检查失败：打开 {dataset_result_json} 文件失败.")
            pass_flag = False

        if not json_data.get("check_pass", False):
            check_dataset_json_message.append("检查失败：数据校验未通过")
            pass_flag = False
            err_type = json_data.get("err_type", None)
            err_msg = json_data.get("err_msg", None)
            if err_type and err_msg:
                check_dataset_json_message.append(f"报错类型：{err_type}")
                check_dataset_json_message.append(f"报错信息：{err_msg}")
            else:
                check_dataset_json_message.append("检查失败：未正确返回报错信息")
        # 检查config和visualdl_log字段对应的文件是否存在
        dataset_path = json_data.get("dataset_path")
        # if not os.path.exists(os.path.join(output_dir, dataset_path)):
        if not os.path.exists(dataset_path):
            check_dataset_json_message.append(f"检查失败：数据集路径 {dataset_path} 不存在")
            pass_flag = False
        if "ts" in module_name:
            show_type = json_data.get("show_type")
            if show_type not in ["csv"]:
                check_dataset_json_message.append(f"检查失败：{show_type} 必须为'csv'")
                pass_flag = False
            for tag in ["train", "val", "test"]:
                samples_key = f"{tag}_table"
                samples_list = json_data["attributes"].get(samples_key)
                if tag == "test" and not samples_list:
                    continue
                if len(samples_list) == 0:
                    check_dataset_json_message.append(f"检查失败：在 {samples_key} 中，值为空")
                    pass_flag = False
        else:
            show_type = json_data.get("show_type")
            if show_type not in ["image", "txt", "csv"]:
                check_dataset_json_message.append(f"检查失败：{show_type} 必须为'image', 'txt', 'csv'其中一个")
                pass_flag = False

            if module_name == "general_recognition":
                tag_list = ["train", "gallery", "query"]
            else:
                tag_list = ["train", "val"]

            for tag in tag_list:
                samples_key = f"{tag}_sample_paths"
                samples_path = json_data["attributes"].get(samples_key)
                for sample_path in samples_path:
                    sample_path = os.path.abspath(os.path.join(output_dir, sample_path))
                    if not samples_path or not os.path.exists(sample_path):
                        check_dataset_json_message.append(f"检查失败：在 {samples_key} 中，{sample_path} 对应的文件不存在或为空")
                        pass_flag = False
            if not (
                "text" in module_name
                or "table" in module_name
                or "formula" in module_name
                or module_name == "general_recognition"
            ):
                try:
                    num_class = int(json_data["attributes"].get("num_classes"))
                except ValueError:
                    check_dataset_json_message.append(f"检查失败：{num_class} 为空或不为整数")
                    pass_flag = False
            if "table" not in module_name:
                analyse_path = json_data["analysis"].get("histogram")
                if not analyse_path or not os.path.exists(os.path.join(output_dir, analyse_path)):
                    check_dataset_json_message.append(f"检查失败：{analyse_path} 数据分析文件不存在或为空")
                    pass_flag = False

        return pass_flag, check_dataset_json_message

    def check_eval_json_content(self, module_name, eval_result_json, check_eval_json_message):
        """Check eval result json content"""
        pass_flag = True
        if not os.path.exists(eval_result_json):
            check_eval_json_message.append(f"检查失败：{eval_result_json} 不存在.")
            pass_flag = False

        try:
            with open(eval_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            check_eval_json_message.append(f"打开 {eval_result_json} 文件失败.")
            pass_flag = False

        if not json_data.get("done_flag", False):
            check_eval_json_message.append("检查失败：评估未完成")
            pass_flag = False
            err_type = json_data.get("err_type", None)
            err_msg = json_data.get("err_msg", None)
            if err_type and err_msg:
                check_eval_json_message.append(f"报错类型：{err_type}")
                check_eval_json_message.append(f"报错信息：{err_msg}")
            else:
                check_eval_json_message.append("检查失败：未正确返回报错信息")

        return pass_flag, check_eval_json_message

    def check_split_dataset(self, output_dir, args_dict, check_splitdata_message):
        """Check the split of a dataset"""
        pass_flag = True
        dst_dataset_path = args_dict["dst_dataset_name"]
        if not os.path.exists(os.path.join(output_dir, dst_dataset_path)):
            check_splitdata_message.append(f"数据划分检查失败：数据集 {dst_dataset_path} 不存在")
            pass_flag = False
            return pass_flag, check_splitdata_message
        if not args_dict.get("split", False):
            check_splitdata_message.append("数据划分检查失败：数据集未划分")
            return pass_flag, check_splitdata_message
        split_dict = {}
        split_train_percent = int(args_dict.get("split_train_percent", 80))
        split_val_percent = int(args_dict.get("split_val_percent", 20))
        split_test_percent = int(args_dict.get("split_test_percent", 0))
        for tag in ["train", "val"]:
            with open(os.path.join(output_dir, dst_dataset_path, "annotations", f"instance_{tag}.json"), "r") as file:
                coco_data = json.load(file)
                split_dict[f"{tag}_nums"] = len(coco_data["images"])
        if split_test_percent == 0:
            try:
                if round(self.process_number(split_dict["train_nums"] / split_dict["val_nums"])) != round(
                    self.process_number(split_train_percent / split_val_percent)
                ):
                    check_splitdata_message.append("数据划分检查失败：数据集划分比例与设定比例不符")
                    pass_flag = False
            except ZeroDivisionError:
                check_splitdata_message.append("数据划分检查失败：split_val_percent 不可设置为0")
                pass_flag = False
        else:
            try:
                if round(self.process_number(split_dict["train_nums"] / split_dict["val_nums"])) != round(
                    self.process_number(split_train_percent / split_val_percent)
                ):
                    check_splitdata_message.append("数据划分检查失败：数据集划分比例与设定比例不符")
                    pass_flag = False
                if round(self.process_number((split_dict["train_nums"] / split_dict["test_nums"]))) != round(
                    self.process_number(split_train_percent / split_test_percent)
                ):
                    check_splitdata_message.append("数据划分检查失败：数据集划分比例与设定比例不符")
                    pass_flag = False
            except ZeroDivisionError:
                check_splitdata_message.append("split_train_percent 和 split_val_percent 不可设置为0")
                pass_flag = False

        return pass_flag, check_splitdata_message

    def process_number(self, num):
        """Process number to avoid division by zero error."""
        if num == 0:
            return "Error: Cannot process zero."
        elif num < 1:
            return 1 / num
        else:
            return num

    def remove_trailing_slash(self, path):
        """Remove trailing slash from the given path."""
        if path.endswith("/"):
            return path[:-1]
        return path

    def run_checks(self, args):
        """Run all checks on the specified arguments"""
        output_dir = args.output
        module_name = args.module_name
        if args.check_dataset_result:
            # 检查 check_result.json 内容
            dataset_result_json = os.path.join(output_dir, "check_dataset_result.json")
            check_dataset_json_message = []
            check_dataset_json_falg, check_dataset_json_message = self.check_dataset_json_content(
                output_dir, module_name, dataset_result_json, check_dataset_json_message
            )
            self.check_results = self.check_results + check_dataset_json_message
            self.check_flag.append(check_dataset_json_falg)

        if args.check_train_result_json:
            # 检查 train_result.json 内容
            train_result_json = os.path.join(output_dir, "train_results.json")
            check_weights_items = args.check_weights_items
            check_train_json_message = []
            check_train_json_flag, check_train_json_message = self.check_train_json_content(
                output_dir, module_name, check_weights_items, train_result_json, check_train_json_message
            )
            self.check_results = self.check_results + check_train_json_message
            self.check_flag.append(check_train_json_flag)

        if args.check_split_dataset:
            # 检查数据划分是否正确
            check_splitdata_message = []
            dataset_result_json = os.path.join(output_dir, "check_dataset_result.json")
            args_dict = json.load(open(dataset_result_json, "r"))
            check_splitdata_flag, check_splitdata_message = self.check_split_dataset(
                output_dir, args_dict, check_splitdata_message
            )
            self.check_results = self.check_results + check_splitdata_message
            self.check_flag.append(check_splitdata_flag)

        if args.check_eval_result_json:
            # 检查 eval_result.json 内容
            eval_result_json = os.path.join(output_dir, "evaluate_result.json")
            check_eval_json_message = []
            check_eval_json_flag, check_eval_json_message = self.check_eval_json_content(
                module_name, eval_result_json, check_eval_json_message
            )
            self.check_results = self.check_results + check_eval_json_message
            self.check_flag.append(check_eval_json_flag)

        assert False not in self.check_flag, print("校验检查失败，请查看产出", " ".join(str(item) for item in self.check_results))
        # print("&&&&&&&&&&&&&", self.check_flag, "!!!!!!!!!!!!!!!!!!!!!!", self.check_results)


if __name__ == "__main__":
    check_items, args = parse_args()
    if args.check:
        checker = PostTrainingChecker(args)
        checker.run_checks(args)
    elif args.download_dataset:
        download_dataset(args)
