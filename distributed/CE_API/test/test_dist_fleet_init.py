#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_fleet_init.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test all api"""

    def test_dist_fleet_init(self):
        """test_dist_fleet_init"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id fleet_init dist_fleet_init.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_init_collective(self):
        """test_dist_fleet_init_collective"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id fleet_init_collective \
        dist_fleet_init_collective.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_init_role(self):
        """test_dist_fleet_init_role"""
        cmd = "python -m paddle.distributed.launch --devices 0 --job_id fleet_init_role dist_fleet_init_role.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_dist_fleet_init_strategy(self):
        """test_dist_fleet_init_strategy"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id fleet_init_strategy \
        dist_fleet_init_strategy.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_dist_fleet_qat_init(self):
    #     """test_dist_fleet_qat_init"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id qat_init \
    #     dist_fleet_qat_init.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1
