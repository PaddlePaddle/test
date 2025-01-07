#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
模型OrderedDict处理工具
"""
import os
import traceback
import numpy as np

from pltools.res_save import save_pickle, load_pickle

if "paddle" in os.environ.get("FRAMEWORK"):
    import paddle

    # import layerApicase
    # import layercase

    if os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleOCR":
        import layerOCRcase
        import PaddleOCR
    elif os.environ.get("USE_PADDLE_MODEL", "None") == "PaddleNLP":
        import layerNLPcase
        import paddlenlp

if "torch" in os.environ.get("FRAMEWORK"):
    import torch

    # import torch_case


class OrderedDictProcess(object):
    """
    用于处理OrderedDict
    """

    # def __init__(self, testing, layerfile, device_id):
    def __init__(self, net, layerfile, orderdict_usage):
        """
        初始化
        """
        self.net = net
        self.orderdict_usage = orderdict_usage
        self.layerfile = layerfile
        # self.modelpath = self.layerfile.replace(".py", "").rsplit(".", 1)[0].replace(".", "/")
        self.layername = self.layerfile.replace(".py", "").rsplit(".", 1)[1].replace(".", "/")

        if isinstance(self.net, torch.nn.Module):
            self.framework = "torch"
            self.modelpath = (
                self.layerfile.replace(".py", "").rsplit(".", 1)[0].replace(".", "/").replace("torch_case/", "")
            )
        elif isinstance(self.net, paddle.nn.Layer):
            self.framework = "paddle"
            self.modelpath = self.layerfile.replace(".py", "").rsplit(".", 1)[0].replace(".", "/")
        else:
            raise ValueError("Unknown framework model in OrderedDictProcess")

        # self.path = os.path.join(os.getcwd(), "orderdict_save", self.framework, self.modelpath, self.layername)
        # os.makedirs(os.path.join(os.getcwd(), "orderdict_save", self.framework, self.modelpath), exist_ok=True)

        self.path = os.path.join(os.getcwd(), "orderdict_save", self.modelpath, self.layername)
        os.makedirs(os.path.join(os.getcwd(), "orderdict_save", self.modelpath), exist_ok=True)

    def save_ordered_dict(self):
        """
        保存OrderedDict到文件
        """
        pickle_dict = {}
        # print('self.net.state_dict() is: ', self.net.state_dict())
        if self.framework == "paddle":
            for key, value in self.net.state_dict().items():
                pickle_dict[key] = value.numpy()
                # print('save pickle_dict[key] is: ', pickle_dict[key])
        elif self.framework == "torch":
            for key, value in self.net.state_dict().items():
                value = value.cpu()
                pickle_dict[key] = value.detach().numpy()
        save_pickle(pickle_dict, self.path)
        # eval(f"{self.framework}.save")(self.net.state_dict(), self.path)

    def load_ordered_dict(self):
        """
        加载文件中的OrderedDict
        """
        ordered_dict = {}
        loaded_data = load_pickle(self.path + ".pickle")
        if self.framework == "paddle":
            for key, value in loaded_data.items():
                ordered_dict[key] = paddle.to_tensor(value)
                # print('load ordered_dict[key] is: ', ordered_dict[key])
            # save_pickle(pickle_dict, self.path)
        elif self.framework == "torch":
            for key, value in loaded_data.items():
                ordered_dict[key] = torch.tensor(value)
                # print('load ordered_dict[key] is: ', ordered_dict[key])
        # ordered_dict = eval(f"{self.framework}.load")(self.path)
        # print('ordered_dict is: ', ordered_dict)
        return ordered_dict

    def set_ordered_dict(self):
        """
        保存OrderedDict到文件
        :param path: 路径
        """
        loaded_ordered_dict = self.load_ordered_dict()
        # net_ordered_dict = self.net.state_dict()
        # for key, value in loaded_ordered_dict.items():
        #     net_ordered_dict[key] = paddle.to_tensor(value.numpy())

        if self.framework == "torch":
            self.net.load_state_dict(loaded_ordered_dict)
        elif self.framework == "paddle":
            self.net.set_state_dict(loaded_ordered_dict)

        return self.net

    def process(self):
        """
        处理OrderedDict
        """
        if self.orderdict_usage == "save":
            self.save_ordered_dict()
        elif self.orderdict_usage == "load":
            self.net = self.set_ordered_dict()
        return self.net


if __name__ == "__main__":

    class LayerCase(paddle.nn.Layer):
        """
        demo
        """

        def __init__(self):
            super().__init__()
            self.conv = paddle.nn.Conv2D(
                in_channels=1, out_channels=1, kernel_size=3, padding=0, weight_attr=None, bias_attr=None
            )

        def forward(self, x):
            """
            forward
            """
            return self.conv(x)

    # class LayerCase(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)

    #     def forward(self, x):
    #         return self.conv(x)

    net = LayerCase()
    odp = OrderedDictProcess(net, "layerApicase.nn_sublayer.Conv1D_2_class")
    odp.save_ordered_dict()
