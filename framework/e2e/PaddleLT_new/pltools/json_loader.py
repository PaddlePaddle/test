#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
加载json
"""

import json


class JSONLoader(object):
    """
    json_loader
    """

    def __init__(self, file_path):
        """initialize"""
        self.file_path = file_path

    def json_dict(self):
        """返回单个json中的dict"""
        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"JSON file '{self.file_path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON file '{self.file_path}'.")
            return None
