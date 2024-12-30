#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
eval 方法
"""
import os
import traceback
import numpy as np
import torch
from engine.torch_xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData

from strategy.ordered_dict import OrderedDictProcess


class TorchLayerEval(object):
    """
    构建Layer评估的通用类
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, testing, layerfile, device_place_id, upstream_net, orderdict_usage="None"):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)

        self.device = os.environ.get("PLT_SET_DEVICE")
        torch.device(f"cuda:{device_place_id}")

        self.testing = testing
        self.upstream_net = upstream_net
        self.orderdict_usage = orderdict_usage
        self.return_net_instance = self.testing.get("return_net_instance", "False")
        self.model_dtype = self.testing.get("model_dtype")
        # torch.set_default_dtype(self.model_dtype) # torch不支持字符串dtype, 测试框架暂时没兼容

        self.layerfile = layerfile
        # self.data = BuildData(layerfile=self.layerfile).get_single_data()

    def _net_input(self):
        """get input"""
        reset(self.seed)
        data = BuildData(layerfile=self.layerfile).get_single_data(framework="torch")
        return data

    def _net_instant(self):
        """get net"""
        reset(self.seed)
        if self.upstream_net:
            net = self.upstream_net
        else:
            net = BuildLayer(layerfile=self.layerfile).get_layer()
        if self.orderdict_usage != "None":
            net = OrderedDictProcess(net=net, layerfile=self.layerfile, orderdict_usage=self.orderdict_usage).process()
        return net

    def torch_dy_eval(self):
        """dygraph eval"""
        net = self._net_instant()
        net.eval()
        logit = net(*self._net_input())
        if self.return_net_instance == "True":
            return {"res": {"logit": logit}, "net": net}
        else:
            return {"res": {"logit": logit}, "net": None}

    # def dy2st_eval(self):
    #     """dy2st eval"""
    #     net = self._net_instant()
    #     st_net = paddle.jit.to_static(net, full_graph=True)
    #     # net.eval()
    #     logit = st_net(*self.data)
    #     return {"logit": logit}

    # def dy2st_eval_cinn(self):
    #     """dy2st eval"""
    #     net = self._net_instant()

    #     build_strategy = paddle.static.BuildStrategy()
    #     build_strategy.build_cinn_pass = True
    #     cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
    #     # net.eval()
    #     logit = cinn_net(*self.data)
    #     return {"logit": logit}
