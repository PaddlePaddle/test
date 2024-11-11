#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_auto_api.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:10
  * @brief
  * disable
  **************************************************************************/
"""
import os
import subprocess

from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestDistAutoApi(object):
    """TestDistAutoApi"""

    # def test_auto_process_mesh(self):
    #     """test_auto_process_mesh"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_process_mesh dist_auto_process_mesh.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    # def test_auto_process_mesh_v2(self):
    #     """test_auto_process_mesh"""
    #     cmd = (
    #         "python -m paddle.distributed.launch --devices 0,1 --job_id auto_process_mesh dist_auto_process_mesh_v2.py"
    #     )
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    # def test_auto_shard_op(self):
    #     """test_auto_shard_op"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_shard_op dist_auto_shard_op.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    # def test_auto_shard_tensor(self):
    #     """test_auto_shard_tensor"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_shard_tensor dist_auto_shard_tensor.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    def test_auto_converter(self):
        """test_auto_converter"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_converter dist_auto_converter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_auto_new_cost_model(self):
    #     """test_auto_new_cost_model"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_new_cost_model \
    #          dist_auto_new_cost_model.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    def test_auto_comm_cost(self):
        """test_auto_comm_cost"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_comm_cost dist_auto_comm_cost.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_auto_comp_cost(self):
        """test_auto_comp_cost"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_comp_cost dist_auto_comp_cost.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_auto_base_cost(self):
    #     """test_auto_base_cost"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id auto_base_cost dist_auto_base_cost.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1
