#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
jit 方法
"""
import os
import numpy as np
import paddle
from engine.paddle_xtools import reset
from generator.builder_layer import BuildLayer
from generator.builder_data import BuildData

from pltools.logger import Logger


class LayerExport(object):
    """
    构建Layer导出的通用类
    """

    def __init__(self, testing, layerfile, device_place_id, upstream_net, orderdict_usage="None"):
        """
        初始化
        """
        self.seed = 33
        reset(self.seed)
        self.device = os.environ.get("PLT_SET_DEVICE")
        paddle.set_device(f"{self.device}:{device_place_id}")

        self.testing = testing
        self.upstream_net = upstream_net
        # self.return_net_instance = self.testing.get("return_net_instance", "False")
        self.model_dtype = self.testing.get("model_dtype")
        paddle.set_default_dtype(self.model_dtype)

        self.layerfile = layerfile
        self.modelpath = self.layerfile.replace(".py", "").rsplit(".", 1)[0].replace(".", "/")
        self.layername = self.layerfile.replace(".py", "").rsplit(".", 1)[1].replace(".", "/")

        self.path = os.path.join(os.getcwd(), "jit_save_export", self.modelpath)

    def _net_input(self):
        """get input"""
        reset(self.seed)
        data = BuildData(layerfile=self.layerfile).get_single_data()
        return data

    def _net_instant(self):
        """get net"""
        reset(self.seed)
        if self.upstream_net:
            net = self.upstream_net
        else:
            net = BuildLayer(layerfile=self.layerfile).get_layer()
        return net

    def _net_input_and_spec(self):
        """get input and inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_spec()
        return data, input_spec

    def _net_input_and_static_spec(self):
        """get input and static inputspec"""
        reset(self.seed)
        data, input_spec = BuildData(layerfile=self.layerfile).get_single_input_and_static_spec()
        return data, input_spec

    def _net_input_and_multi_spec(self):
        """get input and multi inputspec"""
        reset(self.seed)
        data, spec_gen = BuildData(layerfile=self.layerfile).get_single_input_and_multi_spec()
        return data, spec_gen

    def jit_save(self):
        """jit.save(layer)"""
        st_net = paddle.jit.to_static(self._net_instant())
        st_net.eval()
        st_net(*self._net_input())

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(st_net, path=os.path.join(self.path, self.layername, "jit_save"))
        return {"res": None}

    def jit_save_inputspec(self):
        """jit.save(layer)"""
        data, input_spec = self._net_input_and_spec()
        Logger("jit_save_inputspec").get_log().info(f"待测动态InputSpec为: {input_spec}")

        net = self._net_instant()
        st_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)
        st_net.eval()
        # st_net(*self._net_input())

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(st_net, path=os.path.join(self.path, self.layername, "jit_save_inputspec"))
        return {"res": None}

    def jit_save_static_inputspec(self):
        """jit.save(layer)"""
        data, input_spec = self._net_input_and_static_spec()
        Logger("jit_save_static_inputspec").get_log().info(f"待测静态InputSpec为: {input_spec}")

        net = self._net_instant()
        st_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)
        st_net.eval()
        # st_net(*self._net_input())

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(st_net, path=os.path.join(self.path, self.layername, "jit_save_static_inputspec"))
        return {"res": None}

    def jit_save_cinn(self):
        """jit.save(layer)"""
        data = self._net_input()
        net = self._net_instant()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
        cinn_net.eval()
        cinn_net(*data)

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(cinn_net, path=os.path.join(self.path, self.layername, "jit_save_cinn"))
        return {"res": None}

    def jit_save_cinn_inputspec(self):
        """jit.save(layer)"""
        data, input_spec = self._net_input_and_spec()
        Logger("jit_save_cinn_inputspec").get_log().info(f"待测动态InputSpec为: {input_spec}")
        net = self._net_instant()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)
        cinn_net.eval()
        # cinn_net(*self._net_input())

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(cinn_net, path=os.path.join(self.path, self.layername, "jit_save_cinn_inputspec"))
        return {"res": None}

    def jit_save_cinn_static_inputspec(self):
        """jit.save(layer)"""
        data, input_spec = self._net_input_and_static_spec()
        Logger("jit_save_cinn_static_inputspec").get_log().info(f"待测静态InputSpec为: {input_spec}")
        net = self._net_instant()

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.build_cinn_pass = True
        cinn_net = paddle.jit.to_static(net, full_graph=True, input_spec=input_spec)
        cinn_net.eval()
        # cinn_net(*self._net_input())

        # paddle.jit.save(net, path=os.path.join(self.path, self.case))
        paddle.jit.save(cinn_net, path=os.path.join(self.path, self.layername, "jit_save_cinn_static_inputspec"))
        return {"res": None}
