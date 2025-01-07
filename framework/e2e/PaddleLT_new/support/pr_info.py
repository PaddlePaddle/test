#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
获取pr信息
"""

import argparse
import json
import os
import re
import sys

import requests


class PRInfo(object):
    """
    pr信息获取
    """

    def __init__(self, pr_id, title_keyword="CINN"):
        """
        init
        """
        self.pr_id = pr_id
        self.title_keyword = title_keyword

    def get_pr_title(self):
        """
        获取pr标题
        """
        response = requests.get(
            f"https://api.github.com/repos/PaddlePaddle/Paddle/pulls/{self.pr_id}",
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        data = json.loads(response.text)
        title = data["title"]
        return title

    def gen_skip_log(self):
        """
        决定pr是否跳过CI
        """
        title = self.get_pr_title()
        if self.title_keyword in title:
            os.system("echo 1 > pr_title.log")
        else:
            os.system("echo 0 > pr_title.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr_id", type=int, default=0, help="pr号码")
    parser.add_argument("--title_keyword", type=str, default="CINN", help="pr触发CI的关键字")
    args = parser.parse_args()
    reporter = PRInfo(pr_id=args.pr_id, title_keyword=args.title_keyword)
    reporter.gen_skip_log()
