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
import time
import shutil
import logging
import tarfile
import argparse
from pathlib import Path
import requests
import colorlog
from prettytable import PrettyTable
from markdown import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
from yaml import safe_load, dump
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


def parse_args():
    """Parse the arguments"""
    parser = argparse.ArgumentParser()
    # For check urls in doc
    parser.add_argument("--check_url", action="store_true", default=False)
    parser.add_argument("--check_env", action="store_true", default=False)
    parser.add_argument("-d", "--dir", default="./docs", type=str, help="The directory to search for Markdown files.")
    parser.add_argument(
        "-m", "--mode", default="all", choices=["all", "internal", "external"], help="The type of links to check."
    )
    # For save ci result for json
    parser.add_argument("--save_result", action="store_true", default=False)
    parser.add_argument("--successed_cmd", type=str, default="")
    parser.add_argument("--failed_cmd", type=str, default="")
    # For download dataset
    parser.add_argument("--download_dataset", action="store_true", default=False)
    parser.add_argument("--module_name", type=str, default=False)
    parser.add_argument("--config_path", type=str, default=False)
    parser.add_argument("--dataset_url", type=str, default=False)
    # For check paddlex output
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


################################### 下载数据集 ###############################################
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


################################### 检查训练结果 #############################################
class PostTrainingChecker:
    """Post training checker class"""

    def __init__(self, args):
        self.check_flag = True

    def update_fail_flag(self, msg):
        """Update check flag"""
        print(msg)
        self.check_flag = False

    def check_train_json_content(self, output_dir, module_name, check_weights_items, train_result_json):
        """Check train result json content"""
        if not os.path.exists(train_result_json):
            msg = f"train_result.json文件不存在,检查路径为：{train_result_json}"
            self.update_fail_flag(msg)

        try:
            with open(train_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            msg = f"无法解析 {train_result_json} 文件的内容."
            self.update_fail_flag(msg)

        print("*" * 20, "开始检查train_result.json,文件内容如下：", "*" * 20)
        print(json_data)
        print("*" * 60)

        if not json_data.get("done_flag", False):
            msg = "train_result.json文件中done_flag字段值为false,\
                说明训练没有成功,如果dict中有'err_type'和'err_msg'字段,\
                则可以通过查看err_type和err_msg来判断具体原因。\
                如果不是这种情况，建议检查训练是否被异常终止，或代码退出时未正确写入done_flag"
            self.update_fail_flag(msg)
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
                            msg = "train_result.json文件中score字段结果为空"
                            self.update_fail_flag(msg)
                    else:
                        try:
                            file_path = os.path.join(output_dir, last_data.get(file_key))
                        except:
                            file_path = ""
                            self.update_fail_flag(msg)
                        if last_data.get(file_key) == "" or not os.path.exists(file_path):
                            msg = f"检查失败：在训练结果中，{file_key} 对应的文件 {last_data.get(file_key)} 不存在或为空,\
                                对于该模型CI强制检查的key包括：{inspection_item}"
                            self.update_fail_flag(msg)
            else:
                config_path = json_data.get("config")
                visualdl_log_path = json_data.get("visualdl_log")
                label_dict_path = json_data.get("label_dict")
                try:
                    file_path = os.path.join(output_dir, config_path)
                except:
                    file_path = ""
                if not os.path.exists(file_path):
                    msg = f"根据json中的config字段信息，未找到配置文件，请确认配置文件是否存在，配置文件路径为：{os.path.join(output_dir, config_path)}"
                    self.update_fail_flag(msg)
                if not ("text" in module_name or "table" in module_name or "formula" in module_name):
                    try:
                        file_path = os.path.join(output_dir, visualdl_log_path)
                    except:
                        file_path = ""
                    if not os.path.exists(file_path):
                        msg = f"根据json中的visualdl_log字段信息，\
                            未找到VisualDL日志文件，请确认VisualDL日志文件是否存在，\
                            VisualDL日志文件路径为:{output_dir}中的{visualdl_log_path}"
                        self.update_fail_flag(msg)
                try:
                    file_path = os.path.join(output_dir, label_dict_path)
                except:
                    file_path = ""
                if not os.path.exists(file_path):
                    msg = f"根据json中的label_dict字段信息，未找到标签映射文件，\
                        请确认标签映射文件是否存在，标签映射文件路径为:\
                        {output_dir}中的{label_dict_path}"
                    self.update_fail_flag(msg)

                inspection_item = check_weights_items.split(",")[1:]
                last_k = check_weights_items.split(",")[0]
                for i in range(1, int(last_k)):
                    last_key = f"last_{i}"
                    last_data = json_data["models"].get(last_key)

                    for file_key in inspection_item:
                        try:
                            file_path = os.path.join(output_dir, last_data.get(file_key))
                        except:
                            file_path = ""
                        if last_data.get(file_key) == "" or not os.path.exists(file_path):
                            msg = f"检查失败：在第{i}轮的训练结果中，\
                                {file_key} 对应的文件 {last_data.get(file_key)} 不存在或为空,\
                                对于该模型CI强制检查的key包括：{inspection_item}"
                            self.update_fail_flag(msg)

                best_key = "best"
                best_data = json_data["models"].get(best_key)
                if best_data.get("score") == "":
                    msg = "train_result.json文件中score字段结果为空"
                    self.update_fail_flag(msg)
                for file_key in inspection_item:
                    try:
                        file_path = os.path.join(output_dir, best_data.get(file_key))
                    except:
                        file_path = ""
                    if best_data.get(file_key) == "" or not os.path.exists(file_path):
                        msg = f"检查失败：在最佳权重结果中，{file_key} 对应的文件 {file_path} 不存在或为空,对于该模型CI强制检查的key包括：{inspection_item}"
                        self.update_fail_flag(msg)
        return self.check_flag

    def check_dataset_json_content(self, output_dir, module_name, dataset_result_json):
        """Check dataset result json content"""
        if not os.path.exists(dataset_result_json):
            msg = f"check_dataset_result.json文件不存在,检查路径为：{dataset_result_json}"
            self.update_fail_flag(msg)

        try:
            with open(dataset_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            msg = f"无法解析 {dataset_result_json} 文件的内容."
            self.update_fail_flag(msg)

        print("*" * 20, "开始检查check_dataset_result.json,文件内容如下：", "*" * 20)
        print(json_data)
        print("*" * 60)

        if not json_data.get("check_pass", False):
            msg = "检查失败：数据校验未通过,如果dict中有'err_type'和'err_msg'字段,则可以通过查看err_type和err_msg来判断具体原因。如果不是这种情况，建议检查数据集是否符合规范"
            self.update_fail_flag(msg)

        if "ts" in module_name:
            show_type = json_data.get("show_type")
            if show_type not in ["csv"]:
                msg = f"对于TS任务，show_type 必须为'csv'，但是检测到的是 {show_type}"
                self.update_fail_flag(msg)
            for tag in ["train", "val", "test"]:
                samples_key = f"{tag}_table"
                samples_list = json_data["attributes"].get(samples_key)
                if tag == "test" and not samples_list:
                    continue
                if len(samples_list) == 0:
                    msg = f"检查失败：在csv表格中，{samples_key} 列为空"
                    self.update_fail_flag(msg)
        else:
            show_type = json_data.get("show_type")
            if show_type not in ["image", "txt", "csv", "video"]:
                msg = f"对于非TS任务，show_type 必须为'image', 'txt', 'csv','video'其中一个，但是检测到的是 {show_type}"
                self.update_fail_flag(msg)

            if module_name in ["general_recognition", "image_feature"]:
                tag_list = ["train", "gallery", "query"]
            else:
                tag_list = ["train", "val"]

            for tag in tag_list:
                samples_key = f"{tag}_sample_paths"
                samples_path = json_data["attributes"].get(samples_key)
                if samples_path is None:
                    msg = f"检查失败：未能获取到{samples_key}字段"
                    self.update_fail_flag(msg)
                    continue
                for sample_path in samples_path:
                    sample_path = os.path.abspath(os.path.join(output_dir, sample_path))
                    if not samples_path or not os.path.exists(sample_path):
                        msg = f"检查失败：{samples_key}字段对应的文件不存在或为空，文件路径为：{sample_path}"
                        self.update_fail_flag(msg)
            if not (
                "text" in module_name
                or "table" in module_name
                or "formula" in module_name
                or module_name in ["image_feature", "face_feature", "general_recognition"]
            ):
                try:
                    num_class = int(json_data["attributes"].get("num_classes"))
                except ValueError:
                    msg = f"检查失败：{num_class} 为空或不为整数"
                    self.update_fail_flag(msg)
            if "table" not in module_name and module_name != "image_feature" and module_name != "face_feature":
                analyse_path = json_data["analysis"].get("histogram")
                if not analyse_path or not os.path.exists(os.path.join(output_dir, analyse_path)):
                    msg = f"检查失败：{analyse_path} 数据分析文件不存在或为空"
                    self.update_fail_flag(msg)

        return self.check_flag

    def check_eval_json_content(self, module_name, eval_result_json):
        """Check eval result json content"""
        if not os.path.exists(eval_result_json):
            msg = "eval_result.json文件不存在"
            self.update_fail_flag(msg)

        try:
            with open(eval_result_json, "r") as file:
                json_data = json.load(file)
        except json.JSONDecodeError:
            msg = f"无法解析 {eval_result_json} 文件的内容."
            self.update_fail_flag(msg)

        print("*" * 20, "开始检查eval_result.json,文件内容如下：", "*" * 20)
        print(json_data)
        print("*" * 60)

        if not json_data.get("done_flag", False):
            msg = "eval_result.json文件中done_flag字段值为false,\
                如果dict中有'err_type'和'err_msg'字段,\
                则可以通过查看err_type和err_msg来判断具体原因。\
                如果不是这种情况，建议检查评估产出是否正确"
            self.update_fail_flag(msg)

        return self.check_flag

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
        print("=" * 20, "开始执行产出结果检查", "=" * 20)
        if args.check_dataset_result:
            # 检查 check_result.json 内容
            dataset_result_json = os.path.join(output_dir, "check_dataset_result.json")
            self.check_dataset_json_content(output_dir, module_name, dataset_result_json)

        if args.check_train_result_json:
            # 检查 train_result.json 内容
            train_result_json = os.path.join(output_dir, "train_result.json")
            check_weights_items = args.check_weights_items
            self.check_train_json_content(output_dir, module_name, check_weights_items, train_result_json)

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
            self.check_eval_json_content(module_name, eval_result_json)

        if self.check_flag:
            print("=" * 20, "产出结果检查通过", "=" * 20)
        else:
            print("=" * 20, "产出结果检查失败,请根据以上错误信息进行排查修复", "=" * 20)
            exit(1)


################################### 检查文档超链接 ###############################################


def extract_links(markdown_text):
    """Extract links from Markdown text"""
    # 使用BeautifulSoup从HTML中提取链接
    html = markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    real_links = []
    for link in links:
        if link.startswith("#"):
            continue
        real_links.append(link)
    return real_links


def is_valid_link(url):
    """Check whether URL is valid or not"""
    headers = {"User-Agent": "Mozilla/5.0"}
    is_valid = False
    for i in range(10):
        try:
            response = requests.get(url, allow_redirects=True, headers=headers, timeout=5)
            is_valid = True if response.status_code == 200 else False
            break
        except:
            time.sleep(1)
            continue
    return is_valid


def check_internal_link(base_path, link):
    """Check internal link validity"""
    # 如果链接包含锚点(#)，只检查文件部分
    link = link.split("#")[0]
    target_path = (base_path / link).resolve()
    return target_path.exists()


def check_links_in_markdown(file_path, index, total, check_mode):
    """Check links in Markdown file"""
    with open(file_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    links = extract_links(markdown_text)
    invalid_links = []
    links_num = len(links)

    for i, link in enumerate(links):
        print(f"[File schedule {index}/{total}][{i+1}/{links_num}]: {link}")
        if link.startswith("http"):
            if check_mode in ["all", "external"]:
                # 检查外部链接
                if not is_valid_link(link):
                    invalid_links.append(link)
        else:
            if check_mode in ["all", "internal"]:
                # 检查内部链接
                check_result = check_internal_link(Path(file_path).parent, link)
                if not check_result:
                    invalid_links.append(link)

    return invalid_links


def check_all_markdown_files(args):
    """Check all Markdown files"""
    markdown_files = list(Path(args.dir).rglob("*.md"))
    all_invalid_links = {}
    markdown_num = len(markdown_files)

    for i, markdown_file in enumerate(markdown_files):
        invalid_links = check_links_in_markdown(markdown_file, i + 1, markdown_num, check_mode=args.mode)
        if invalid_links:
            all_invalid_links[markdown_file] = invalid_links

    return all_invalid_links


def check_documentation_url(args):
    """Check documentation URL"""
    invalid_links = check_all_markdown_files(args)
    output_file = "invalid_links.txt"

    if len(invalid_links) > 0:
        print("Found invalid links in the documentation files, details are as follows:")
        print("*" * 80)
        for file, links in invalid_links.items():
            print(f"Invalid links in {file}:")
            for link in links:
                print(f"  - {link}")
        print("*" * 80)

        with open(output_file, "w", encoding="utf-8") as f:
            for file, links in invalid_links.items():
                f.write(f"Invalid links in {file}: \n")
                for link in links:
                    f.write(f"  - {link}\n")
        print(f"Invalid links have been saved to {output_file}.")
        exit(1)
    else:
        pass


def get_option(cmd):
    "get_option"
    if "--check" in cmd:
        cmd_list = cmd.split(" ")
        check_option = cmd_list[5]
        model_name = cmd_list[7].split("/")[-1]
        if check_option == "--check_train_result_json":
            option = f"{model_name}_check_train_result_json"
        elif check_option == "--check_eval_result_json":
            option = f"{model_name}_check_eval_result_json"
        elif check_option == "--check_dataset_result":
            option = f"{model_name}_check_dataset_result"
        logfile = cmd_list[-1]
        return option, logfile
    elif "--pipeline" in cmd:
        cmd_list = cmd.split(" ")
        pipeline_name = cmd_list[4]
        logfile = cmd_list[-1]
        return pipeline_name, logfile
    elif "main.py" in cmd:
        cmd_list = cmd.split(" ")
        model_name = cmd_list[5].split("/")[-1].split(".yaml")[0]
        option = cmd_list[7].split("=")[-1]
        if option == "train":
            logfile = cmd_list[-1].split("=")[-1]
            logfile += "train.log"
        else:
            logfile = None
        option = f"{model_name}_{option}"
        return option, logfile
    else:
        return None, None


def save_result_json(args):
    "save_result_json"
    with open(args.successed_cmd, "r") as f:
        successed_cmd_list = f.readlines()
    with open(args.failed_cmd, "r") as f:
        failed_cmd_list = f.readlines()
    result = {"successed": [], "failed": []}
    successed_num = len(successed_cmd_list) - 1
    failed_num = len(failed_cmd_list) - 1
    result["successed_num"] = successed_num
    result["failed_num"] = failed_num
    for i, cmd in enumerate(successed_cmd_list[1:]):
        check_option, logfile = get_option(cmd.strip())
        if check_option:
            result["successed"].append(check_option)
    for i, cmd in enumerate(failed_cmd_list[1:]):
        check_option, logfile = get_option(cmd.strip())
        if check_option:
            try:
                with open(logfile, "r") as f:
                    log_info = f.readlines()
            except:
                log_info = "未成功提取log，请在全局log中自行查找"
            result["failed"].append({check_option: log_info})
    with open("ci/results.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)
    # print(args.failed_cmd)


def get_logger(level=logging.INFO):
    """
    获取带有颜色的logger
    """
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(level)
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # 定义颜色输出格式
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    # 将颜色输出格式添加到控制台日志处理器
    console_handler.setFormatter(color_formatter)
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # 将控制台日志处理器添加到logger对象
    logger.addHandler(console_handler)
    return logger


def format_commit(commit):
    """
    格式化commit信息
    """
    commit = "-".join(commit[i : i + 2] for i in range(0, len(commit), 2))
    return commit


def check_environment():
    """检查环境是否满足要求"""
    table = PrettyTable(["Environment", "Value", "Description"])
    device_type = os.environ.get("DEVICE_TYPE", None)
    device_ids = os.environ.get("DEVICE_ID", None)
    msg = "=" * 20 + "以下为当前环境变量配置信息" + "=" * 20
    logger.info(msg)
    if not device_type and not device_ids:
        logger.warning("未设置设备类型和编号，使用配置文件中的默认值")

    table.add_row(["Device", os.environ.get("DEVICE_TYPE", "GPU"), "算力卡类型，例如GPU、NPU、XPU等，默认为GPU"])
    table.add_row(["Device ID", os.environ.get("DEVICE_ID", "0,1,2,3"), "使用的设备id列表，多个用逗号分隔"])
    table.add_row(["MEM_SIZE", os.environ.get("MEM_SIZE", "32"), "显存大小，单位GB"])
    table.add_row(["TEST_RANGE", os.environ.get("TEST_RANGE", ""), "测试范围，默认为空，设置示例：export TEST_RANGE='inference'"])
    table.add_row(["MD_NUM", os.environ.get("MD_NUM", ""), "PR中MD文件改动数量，用于判断是否需要进行文档超链接检测，默认为空，设置示例：export MD_NUM=10"])
    table.add_row(
        ["WITHOUT_MD_NUM", os.environ.get("WITHOUT_MD_NUM", ""), "PR中除去MD文件改动数量，默认为空，设置示例：export WITHOUT_MD_NUM=10"]
    )
    logger.info(table)
    msg = "=" * 20 + "以下为Paddle相关版本信息" + "=" * 20
    logger.info(msg)
    logger.info("注：为了能够显示commit信息，在每个数字中增加了一个“-”连字符，实际使用请删除连字符")
    table = PrettyTable(["Repository", "Branch", "Commit"])
    try:
        import paddle

        commit = paddle.__git_commit__
        table.add_row(["PaddlePaddle", paddle.__version__, format_commit(commit)])
    except ImportError:
        logger.error("Paddle is not installed,please install it first.")
        exit(1)
    try:
        branch = os.popen("git rev-parse --abbrev-ref HEAD").read().strip()
        commit = os.popen("git rev-parse HEAD").read().strip()
        table.add_row(["PaddleX", branch, format_commit(commit)])
    except ImportError:
        logger.error("PaddleX is not installed,please install it first.")
        exit(1)
    repos = os.listdir("paddlex/repo_manager/repos")
    for repo in repos:
        if repo == ".gitkeep":
            continue
        try:
            branch = os.popen(f"cd paddlex/repo_manager/repos/{repo} && git rev-parse --abbrev-ref HEAD").read().strip()
            commit = os.popen(f"cd paddlex/repo_manager/repos/{repo} &&git rev-parse HEAD").read().strip()
            table.add_row([repo, branch, format_commit(commit)])
        except:
            pass
    _logger = get_logger()
    _logger.info(table)
    msg = "=" * 20 + "环境检测完成" + "=" * 20
    logger.info(msg)


if __name__ == "__main__":
    check_items, args = parse_args()
    logger = get_logger()
    if args.check:
        checker = PostTrainingChecker(args)
        checker.run_checks(args)
    elif args.download_dataset:
        download_dataset(args)
    elif args.check_url:
        check_documentation_url(args)
    elif args.save_result:
        save_result_json(args)
    elif args.check_env:
        check_environment()
