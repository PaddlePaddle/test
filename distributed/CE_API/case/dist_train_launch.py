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
  * @file dist_train_launch.py
  * @author liyang109@baidu.com
  * @date 2020-11-17 16:30
  * @brief
  *
  **************************************************************************/
"""
import os
import numpy as np
import paddle.incubate.distributed.fleet.role_maker as role_maker
from paddle.incubate.distributed.fleet.collective import fleet, DistributedStrategy
import paddle


paddle.enable_static()


def gen_data():
    """generate data"""
    np.random.seed(1)
    return {
        "x": np.random.random(size=(128, 32)).astype("float32"),
        "y": np.random.randint(2, size=(128, 1)).astype("int64"),
    }


def mlp(input_x, input_y, hid_dim=128, label_dim=2):
    """mlp net struct"""
    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim, activation="tanh")
    fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim, activation="tanh")
    prediction = paddle.static.nn.fc(x=[fc_2], size=label_dim, activation="softmax")
    cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction="none", use_softmax=False)
    avg_cost = paddle.mean(x=cost)
    return avg_cost


input_x = paddle.static.data(name="x", shape=[-1, 32], dtype="float32")
input_y = paddle.static.data(name="y", shape=[-1, 1], dtype="int64")

cost = mlp(input_x, input_y)
optimizer = paddle.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(cost)

gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
place = paddle.CUDAPlace(gpu_id)

exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())

step = 5
train_info = []
for i in range(step):
    cost_val = exe.run(program=paddle.static.default_main_program(), feed=gen_data(), fetch_list=[cost])
    train_info.append(cost_val[0])
print(train_info)
