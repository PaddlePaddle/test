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


class SingleImageWithoutLabelDataset(Dataset):
    """Single Image Without Label Dataset, use paddle.io.Dataset"""

    def __init__(self, data_dict, num_samples=1):
        self.data_dict = data_dict
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data_dict = self.data_dict
        return data_dict

    def __len__(self):
        return self.num_samples


class SingleImageWithoutLabel:
    """Single Image Without Label Dataset, not use paddle.io.Dataset"""

    def __init__(self, data_dict, num_samples=1):
        self.data_dict = data_dict
        self.num_samples = num_samples

    def __getitem__(self, idx):
        data_dict = self.data_dict
        return data_dict

    def __len__(self):
        return self.num_samples


class SingleImageWithLabelDataset(Dataset):
    """Single Image With Label Dataset, use paddle.io.Dataset"""

    def __init__(self, image, label, num_samples):
        self.image = image
        self.label = label
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = self.image
        label = self.label
        return image, label

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    np.random.seed(33)
    paddle.seed(33)
    data_dict = {"image": paddle.to_tensor(tool._randtool("float32", -1, 1, shape=[2, 3])), "tag": 24}
    dataset = SingleImageWithoutLabel(data_dict=data_dict, num_samples=12)
    BATCH_SIZE = 2
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=2)
    # for i in range(len(dataset)):
    #     print('step {}: {}'.format(i, dataset[i]))
    for i, img in enumerate(loader()):
        print("step {}: {}".format(i, img))
    from collections import Iterable

    if isinstance(dataset, paddle.io.Dataset):
        print("loader is!!!")
    else:
        print("loader not!!!")
    print("i is: ", i)
    print(dataset[100])

    # np.random.seed(33)
    # in_image = tool._randtool("float32", -1, 1, shape=[2, 3])
    # dataset = SingleImageWithoutLabel(image=in_image, num_samples=12)
    # for i in range(len(dataset)):
    #     print('step {}: {}'.format(i, dataset[i]))
