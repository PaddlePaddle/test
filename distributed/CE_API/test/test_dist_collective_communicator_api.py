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
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_collective_communicator_api.py
  * @author liujie44@baidu.com
  * @date 2021-11-09 11:00
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess

from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """test all collective communicator api"""

    def test_collective_all_gather(self):
        """test_collective_all_gather"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id all_gather dist_collective_all_gather.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_all_gather_object(self):
        """test_collective_all_gather_object"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id all_gather_object \
            dist_collective_all_gather_object.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_all_reduce(self):
        """test_collective_all_reduce"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id all_reduce dist_collective_all_reduce.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_barrier(self):
        """test_collective_barrier"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id barrier dist_collective_barrier.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_broadcast_object_list(self):
        """test_collective_broadcast_object_list"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id broadcast_object_list \
                dist_collective_broadcast_object_list.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_broadcast(self):
        """test_collective_broadcast"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id broadcast dist_collective_broadcast.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_reduce(self):
        """test_collective_reduce"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id reduce dist_collective_reduce.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_reduceop(self):
        """test_reduceop"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id reduceop dist_collective_reduceop.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_scatter(self):
        """test_collective_scatter"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id scatter dist_collective_scatter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_reduce_scatter(self):
        """test_collective_reduce_scatter"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id reduce_scatter \
            dist_collective_reduce_scatter.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    # def test_collective_split(self):
    #     """test_collective_split"""
    #     cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id split dist_collective_split.py"
    #     pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     out, err = pro.communicate()
    #     print(out)
    #     pro.wait()
    #     pro.returncode == 0
    #     assert str(out).find("Error") == -1
    #     assert str(err).find("Error") == -1

    def test_collective_send(self):
        """test_collective_send"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id send dist_collective_send.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_isend(self):
        """test_collective_isend"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id isend dist_collective_isend.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_recv(self):
        """test_collective_recv"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id recv dist_collective_recv.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_irecv(self):
        """test_collective_irecv"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id irecv dist_collective_irecv.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_alltoall(self):
        """test_collective_alltoall"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id alltoall dist_collective_alltoall.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_alltoall_single(self):
        """test_collective_alltoall_single"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id alltoall_single \
            dist_collective_alltoall_single.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_new_group1(self):
        """test_collective_new_group1"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id new_group1 dist_collective_new_group1.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1

    def test_collective_new_group2(self):
        """test_collective_new_group2"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id new_group2 dist_collective_new_group2.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1
