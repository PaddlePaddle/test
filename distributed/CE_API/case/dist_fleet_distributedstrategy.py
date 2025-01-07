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
  * @file dist_fleet_distributedstrategy.py
  * @author liujie44@baidu.com
  * @date 2021-11-12 14:41
  * @brief
  *
  **************************************************************************/
"""
import time
from time import sleep

import paddle
import paddle.distributed.fleet as fleet

from utils import run_priority


@run_priority(level="P0")
def test_dist_fleet_DistributedStrategy1():
    """test_dist_fleet_DistributedStrategy"""
    strategy1 = fleet.DistributedStrategy()

    strategy1.recompute = True
    strategy1.recompute_configs = {"checkpoints": ["x"]}
    strategy1.gradient_merge = True
    strategy1.gradient_merge_configs = {"k_steps": 4, "avg": True}
    strategy1.lamb = True
    strategy1.lamb_configs = {
        "lamb_weight_decay": 0.01,
        "exclude_from_weight_decay": [],
    }
    strategy1.adaptive_localsgd = True
    strategy1.adaptive_localsgd_configs = {"init_k_steps": 1, "begin_step": 30}
    strategy1.dgc = True
    strategy1.dgc_configs = {"rampup_begin_step": 1252}
    strategy1.sharding = True
    strategy1.sharding_configs = {
        "sharding_segment_strategy": "segment_broadcast_MB",
        "segment_broadcast_MB": 32,
        "sharding_degree": 2,
        "gradient_merge_acc_step": 4,
    }

    strategy1.save_to_prototxt("origin_dist_strategy1.prototxt")

    new_strategy = fleet.DistributedStrategy()
    new_strategy.load_from_prototxt("origin_dist_strategy1.prototxt")

    time.sleep(30)
    assert new_strategy.recompute == strategy1.recompute
    assert new_strategy.recompute_configs == strategy1.recompute_configs
    assert new_strategy.gradient_merge == strategy1.gradient_merge
    assert new_strategy.gradient_merge_configs == strategy1.gradient_merge_configs
    assert new_strategy.lamb == strategy1.lamb
    assert new_strategy.lamb_configs == strategy1.lamb_configs
    assert new_strategy.adaptive_localsgd == strategy1.adaptive_localsgd
    assert new_strategy.adaptive_localsgd_configs == strategy1.adaptive_localsgd_configs
    assert new_strategy.dgc == strategy1.dgc
    assert new_strategy.dgc_configs == strategy1.dgc_configs
    assert new_strategy.sharding == strategy1.sharding
    assert new_strategy.sharding_configs == strategy1.sharding_configs

    assert new_strategy.auto is False
    assert new_strategy.pipeline is False
    assert new_strategy.lars is False
    assert new_strategy.localsgd is False
    assert new_strategy.amp is False
    assert new_strategy.fp16_allreduce is False

    print("test_dist_fleet_DistributedStrategy1 ... ok")


@run_priority(level="P0")
def test_dist_fleet_DistributedStrategy2():
    """test_dist_fleet_DistributedStrategy"""
    strategy2 = fleet.DistributedStrategy()

    strategy2.auto = True
    strategy2.pipeline = True
    strategy2.pipeline_configs = {"micro_batch_size": 12}
    strategy2.lars = True
    strategy2.lars_configs = {
        "lars_coeff": 0.001,
        "lars_weight_decay": 0.0005,
        "epsilon": 0,
        "exclude_from_weight_decay": ["batch_norm", ".b"],
    }
    strategy2.localsgd = True
    strategy2.localsgd_configs = {"k_steps": 4, "begin_step": 30}
    strategy2.amp = True
    strategy2.amp_configs = {"init_loss_scaling": 32768, "custom_white_list": ["conv2d"]}
    strategy2.fp16_allreduce = True
    strategy2.a_sync = True

    strategy2.save_to_prototxt("origin_dist_strategy2.prototxt")

    new_strategy = fleet.DistributedStrategy()
    new_strategy.load_from_prototxt("origin_dist_strategy2.prototxt")

    time.sleep(30)
    assert new_strategy.auto == strategy2.auto
    assert new_strategy.pipeline == strategy2.pipeline
    assert new_strategy.pipeline_configs == strategy2.pipeline_configs
    assert new_strategy.lars == strategy2.lars
    assert new_strategy.lars_configs == strategy2.lars_configs
    assert new_strategy.localsgd == strategy2.localsgd
    assert new_strategy.localsgd_configs == strategy2.localsgd_configs
    assert new_strategy.amp == strategy2.amp
    assert new_strategy.amp_configs == strategy2.amp_configs
    assert new_strategy.fp16_allreduce == strategy2.fp16_allreduce
    assert new_strategy.a_sync == strategy2.a_sync

    assert new_strategy.recompute is False
    assert new_strategy.gradient_merge is False
    assert new_strategy.lamb is False
    assert new_strategy.adaptive_localsgd is False
    assert new_strategy.dgc is False
    assert new_strategy.sharding is False

    print("test_dist_fleet_DistributedStrategy2 ... ok")


@run_priority(level="P0")
def test_dist_fleet_DistributedStrategy3():
    """test build_strategy"""
    strategy = fleet.DistributedStrategy()

    build_strategy = paddle.static.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_act_ops = True
    build_strategy.enable_auto_fusion = True
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_all_optimizer_ops = True
    build_strategy.enable_inplace = True

    strategy.build_strategy = build_strategy

    strategy.save_to_prototxt("origin_dist_strategy.prototxt")
    new_strategy = fleet.DistributedStrategy()
    new_strategy.load_from_prototxt("origin_dist_strategy.prototxt")

    time.sleep(30)
    assert new_strategy.build_strategy.fuse_broadcast_ops == build_strategy.fuse_broadcast_ops
    assert new_strategy.build_strategy.enable_inplace == build_strategy.enable_inplace
    assert new_strategy.build_strategy.cache_runtime_context is False
    assert new_strategy.build_strategy.enable_addto is False
    assert new_strategy.build_strategy.allow_cuda_graph_capture is False

    print("test_dist_fleet_DistributedStrategy4 ... ok")


if __name__ == "__main__":
    test_dist_fleet_DistributedStrategy1()
    test_dist_fleet_DistributedStrategy2()
    test_dist_fleet_DistributedStrategy3()
