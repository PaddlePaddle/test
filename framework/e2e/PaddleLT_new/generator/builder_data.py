#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
data builder
"""

import os
import itertools
import numpy as np

if "paddle" in os.environ.get("FRAMEWORK"):
    import paddle
    import diy
    import layerApicase
    import layercase

    if os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleOCR":
        import layerOCRcase
        import PaddleOCR
    elif os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleNLP":
        import layerNLPcase
        import paddlenlp

if "torch" in os.environ.get("FRAMEWORK"):
    import torch
    import torch_case

import pltools.np_tool as tool


class BuildData(object):
    """BuildData"""

    def __init__(self, layerfile):
        """init"""
        self.layerfile = layerfile
        self.layer_module = eval(self.layerfile)

    def get_single_data(self, framework="paddle"):
        """get data"""
        if hasattr(self.layer_module, "create_numpy_inputs"):
            # dataname = self.layerfile + ".create_numpy_inputs()"
            data = []
            # for i in eval(dataname):
            for i in getattr(self.layer_module, "create_numpy_inputs")():
                if isinstance(i, (tuple, list)):  # 为了适配list输入的模型子图
                    tmp = []
                    for j in i:
                        if framework == "paddle":
                            if j.dtype == np.int64 or j.dtype == np.int32:
                                tmp.append(paddle.to_tensor(j, stop_gradient=True))
                            else:
                                tmp.append(paddle.to_tensor(j, stop_gradient=False))
                        elif framework == "torch":
                            if j.dtype == np.int64 or j.dtype == np.int32:
                                # tmp.append(torch.tensor(j, requires_grad=False, device=torch.device('cuda:0')))
                                tmp.append(torch.tensor(j, requires_grad=False))
                            else:
                                # tmp.append(torch.tensor(j, requires_grad=True, device=torch.device('cuda:0')))
                                tmp.append(torch.tensor(j, requires_grad=True))
                    data.append(tmp)
                elif isinstance(i, np.ndarray):
                    if framework == "paddle":
                        if i.dtype == np.int64 or i.dtype == np.int32:
                            data.append(paddle.to_tensor(i, stop_gradient=True))
                        else:
                            data.append(paddle.to_tensor(i, stop_gradient=False))
                    elif framework == "torch":
                        if i.dtype == np.int64 or i.dtype == np.int32:
                            # data.append(torch.tensor(i, requires_grad=False, device=torch.device('cuda:0')))
                            data.append(torch.tensor(i, requires_grad=False))
                        else:
                            # data.append(torch.tensor(i, requires_grad=True, device=torch.device('cuda:0')))
                            data.append(torch.tensor(i, requires_grad=True))
                elif isinstance(i, float):
                    data.append(paddle.to_tensor(i, stop_gradient=False))
                elif isinstance(i, int):
                    data.append(paddle.to_tensor(i, stop_gradient=True))
                else:
                    data.append(i)
        else:
            data = self.get_single_tensor()
        return data

    def get_single_tensor(self):
        """get data"""
        # dataname = self.layerfile + ".create_tensor_inputs()"
        data = []
        # for i in eval(dataname):
        for i in getattr(self.layer_module, "create_tensor_inputs")():
            data.append(i)

        return data

    def get_single_numpy(self):
        """get data"""
        # dataname = self.layerfile + ".create_numpy_inputs()"
        data = []
        # for i in eval(dataname):
        for i in getattr(self.layer_module, "create_numpy_inputs")():
            data.append(i)

        return data

    def get_single_input_and_static_spec(self):
        """get single inputspec"""
        spec_list = []
        data = self.get_single_data()
        for v in data:
            if isinstance(v, paddle.Tensor):
                spec_tmp = paddle.static.InputSpec(
                    shape=v.shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
                )
                spec_list.append(spec_tmp)
        return data, spec_list

    def get_single_input_and_dynamic_spec(self):
        """get single inputspec"""
        spec_list = []
        data = self.get_single_data()
        for v in data:
            if isinstance(v, paddle.Tensor):
                input_shape = tuple([-1] * len(v.shape))
                spec_tmp = paddle.static.InputSpec(
                    shape=input_shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
                )
                spec_list.append(spec_tmp)
        return data, spec_list

    def get_single_input_and_spec(self):
        """get single inputspec"""
        spec_list = []
        data = self.get_single_data()
        if hasattr(eval(self.layerfile), "create_inputspec"):  # 如果子图case中包含inputspec, 则直接使用接口获取
            spec_list = eval(self.layerfile + ".create_inputspec()")
        else:
            for v in data:
                if isinstance(v, paddle.Tensor):
                    spec_tmp = paddle.static.InputSpec(
                        shape=v.shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
                    )
                    spec_list.append(spec_tmp)
        return data, spec_list

    def get_single_input_and_multi_spec(self):
        """get single inputspec"""
        inputs_info = []
        data = self.get_single_data()
        for v in data:
            if isinstance(v, paddle.Tensor):
                spec_tmp = SpecInfoMeta(shape=v.shape, dtype=v.dtype, stop_gradient=v.stop_gradient)
                inputs_info.append(spec_tmp)
        spec_gen = SpecStrategy(inputs_info)
        return data, spec_gen

    # def get_single_input_and_multi_spec_legacy(self):
    #     """get single inputspec"""
    #     spec_all_list = []
    #     data = self.get_single_data()
    #     for index in len(data):
    #         spec_list = []
    #         maybe_shapes = self._input_spec_shape_search(data[index])
    #         for i, v in enumerate(data):
    #             for j, w in enumerate(maybe_shapes):
    #                 if isinstance(v, paddle.Tensor):
    #                     if i == index:
    #                         input_shape = w
    #                     else:
    #                         input_shape = tuple([-1] * len(v.shape))
    #                     spec_tmp = paddle.static.InputSpec(
    #                         shape=input_shape, dtype=v.dtype, name=None, stop_gradient=v.stop_gradient
    #                     )
    #                     spec_list.append(spec_tmp)
    #         spec_all_list.append(spec_list)
    #     return data, spec_all_list

    # def _input_spec_shape_search(self, shape):
    #     """用于搜索inputspect可能的shape组合"""
    #     shape_eles = [[-1, s] for s in shape]
    #     raw_shapes = itertools.product(*shape_eles)
    #     has_channel = len(shape) == 4 and shape[1] == 3

    #     maybe_shapes = []
    #     for s in raw_shapes:
    #         # in case of channel
    #         if has_channel and s[1] is None:
    #             continue
    #         maybe_shapes.append(s)
    #     return maybe_shapes


# 用于动态InputSpec的遍历搜索
class SpecInfoMeta:
    """
    构建可能的InputSpec
    """

    def __init__(self, shape, dtype, stop_gradient):
        self.shape = shape
        self.dtype = dtype
        self.stop_gradient = stop_gradient

    def as_spec(self, shape):
        """
        as_spec
        """
        return paddle.static.InputSpec(shape=shape, dtype=self.dtype, stop_gradient=self.stop_gradient)

    def maybe_shapes(self):
        """
        找到单个InputSpec中, 所有可能的shape
        """
        shape_eles = [[-1, s] for s in self.shape]
        raw_shapes = itertools.product(*shape_eles)
        has_channel = len(self.shape) == 4 and self.shape[1] == 3

        maybe_shapes = []
        for shape in raw_shapes:
            # in case of channel
            if has_channel and shape[1] is None:
                continue
            maybe_shapes.append(shape)
        return maybe_shapes

    def maybe_specs(self):
        """
        maybe_specs
        """
        specs = []
        for shape in self.maybe_shapes():
            specs.append(self.as_spec(shape))
        return specs


class SpecStrategy:
    """
    SpecStrategy生成器
    """

    def __init__(self, inputs_info):
        """
        inputs_info: 是一个list, 包含多个SpecInfoMeta对象, 具体形式为[SpecInfoMeta(shape, dtype, stop_gradient), ...]
        """
        self.inputs_info = inputs_info

    def next(self):
        """
        next
        """
        strategy = self.parse_strategy()
        for specs in itertools.product(*strategy):
            yield specs

    def parse_strategy(self):
        """
        生成方法
        """
        strategy = []
        for input_info in self.inputs_info:
            strategy.append(input_info.maybe_specs())
        return strategy
