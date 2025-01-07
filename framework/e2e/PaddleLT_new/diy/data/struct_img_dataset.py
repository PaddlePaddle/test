#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
single image Paddle Dataset
"""

import numpy as np

import paddle
from paddle.io import Dataset, BatchSampler, DataLoader
import pltools.np_tool as tool


# class ListImageWithoutLabelDataset(Dataset):
#     """List Image Without Label Dataset, use paddle.io.Dataset"""
#
#     def __init__(self, data_dict, num_samples=1):
#         self.data_dict = data_dict
#         for k, v in self.data_dict.items():
#             self.data_dict[k] = [v, v]
#         self.num_samples = num_samples
#
#     def __getitem__(self, idx):
#         data_dict = self.data_dict
#         return data_dict
#
#     def __len__(self):
#         return self.num_samples
#
#
# class ListImageWithoutLabel:
#     """List Image Without Label Dataset, not use paddle.io.Dataset"""
#
#     def __init__(self, data_dict, num_samples=1):
#         self.data_dict = data_dict
#         for k, v in self.data_dict.items():
#             self.data_dict[k] = [v, v]
#         self.num_samples = num_samples
#
#     def __getitem__(self, idx):
#         data_dict = self.data_dict
#         return data_dict
#
#     def __len__(self):
#         return self.num_samples
#
#
# class ListImageWithoutLabel1:
#     """List Image Without Label Dataset, not use paddle.io.Dataset"""
#
#     def __init__(self, data_dict, num_samples=1):
#         self.data_dict = data_dict
#         self.new_data_dict = {}
#         self.new_data_list = []
#         self.new_key_list = []
#         for k, v in self.data_dict.items():
#             self.new_key_list.append(k)
#             self.new_data_list.append(v)
#         self.new_data_dict[self.new_key_list[0]] = self.new_data_list
#         self.num_samples = num_samples
#
#     def __getitem__(self, idx):
#         data_dict = self.new_data_dict
#         return data_dict
#
#     def __len__(self):
#         return self.num_samples


class DictImageWithoutLabel:
    """List Image Without Label Dataset, not use paddle.io.Dataset"""

    def __init__(self, data_dict, num_samples=1):
        self.data_dict = data_dict
        for k, v in self.data_dict.items():
            self.data_dict[k] = {"image": v}
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data_dict = self.data_dict
        return data_dict

    def __len__(self):
        return self.num_samples
