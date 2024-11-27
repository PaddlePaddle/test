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
import tarfile
import argparse
from pathlib import Path
import requests
from markdown import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
from yaml import safe_load, dump
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


def parse_args():
    """Parse the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_url", action="store_true", default=False)
    parser.add_argument("-d", "--dir", default="./docs", type=str, help="The directory to search for Markdown files.")
    parser.add_argument(
        "-m", "--mode", default="all", choices=["all", "internal", "external"], help="The type of links to check."
    )
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
            if show_type not in ["image", "txt", "csv", "video"]:
                check_dataset_json_message.append(f"检查失败：{show_type} 必须为'image', 'txt', 'csv','video'其中一个")
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
                or module_name == "face_feature"
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
            train_result_json = os.path.join(output_dir, "train_result.json")
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


if __name__ == "__main__":
    check_items, args = parse_args()
    if args.check:
        checker = PostTrainingChecker(args)
        checker.run_checks(args)
    elif args.download_dataset:
        download_dataset(args)
    elif args.check_url:
        check_documentation_url(args)
