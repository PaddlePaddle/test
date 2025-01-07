#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
yaml base
"""

import yaml


class YamlLoader(object):
    """
    yaml_loader
    """

    def __init__(self, yml):
        """initialize"""
        try:
            with open(yml, encoding="utf-8") as f:
                self.yml = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print(e)

    def __str__(self):
        """str"""
        return str(self.yml)

    def get_case_info(self, case_name):
        """
        get case info
        """
        return {"info": self.yml.get(case_name), "name": case_name}

    def get_all_case_name(self):
        """
        get all case name
        """
        # 获取全部case name
        if bool(self.yml):
            return self.yml.keys()
        else:
            return []

    def get_junior_name(self, junior):
        """
        get testings name
        """
        # 获取全部case name
        if bool(self.yml):
            return self.yml.get(junior).keys()
        else:
            return []
