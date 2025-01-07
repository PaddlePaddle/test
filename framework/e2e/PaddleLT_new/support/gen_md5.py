#!/bin/env python
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
md5获取
"""

import uuid
import hashlib
import cpuinfo
import psutil
import pynvml


task_name = "CI_paddlelt_train_cinn_eval_cinn_inputspec"
md5 = hashlib.md5()
md5.update((task_name).encode("utf-8"))
# 获取 MD5 值
md5_value = md5.hexdigest()
print(md5_value)
