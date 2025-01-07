#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
# ======================================================================
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""
/***************************************************************************
  *
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file dist_fleet_qat_init.py
  * @author liujie44@baidu.com
  * @date 2024-02-21
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle.nn.functional as F
from utils import run_priority

paddle.enable_static()
fleet.init(is_collective=True)
linear = paddle.nn.Linear(10, 10)
strategy = fleet.DistributedStrategy()
optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=linear.parameters())
optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)


def run_example_code():
    """run_example_code"""
    place = paddle.CUDAPlace(dist.get_rank())
    exe = paddle.static.Executor(place)
    # 1. Define the train program
    data = paddle.static.data(name="X", shape=[None, 1, 28, 28], dtype="float32")
    Conv = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=3)
    conv2d = Conv(data)
    bn = paddle.nn.BatchNorm2D(6)(conv2d)
    pool = F.max_pool2d(bn, kernel_size=2, stride=2)
    hidden = paddle.static.nn.fc(pool, size=10)
    loss = paddle.mean(hidden)
    # 2. Create the distributed optimizer and set qat config to True.
    optimizer = paddle.optimizer.Momentum(learning_rate=0.01, multi_precision=True)
    strategy = fleet.DistributedStrategy()
    strategy.qat = True
    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    # 3. Apply the strategies by distributed optimizer
    # If you don't use the default_startup_program(), you sholud pass
    # your defined `startup_program` into `minimize`.
    optimizer.minimize(loss)
    exe.run(paddle.static.default_startup_program())
    # 4. Use `qat_init` to do FP32 parameters initialization.
    # If you want to perform the testing process, you should pass `test_program` into `qat_init`.
    optimizer.qat_init(place, paddle.static.global_scope())


@run_priority(level="P0")
def test_fleet_qat_init():
    """test_fleet_qat_init"""
    if paddle.is_compiled_with_cuda() and len(paddle.static.cuda_places()) > 0:
        run_example_code()
        exe = paddle.static.Executor(paddle.CPUPlace())
        fleet.save_persistables(exe, "./save_persistables", paddle.static.default_main_program())

    print("test_fleet_qat_init ... ok")
    print("test_fleet_save_persistables ... ok")


if __name__ == "__main__":
    test_fleet_qat_init()
