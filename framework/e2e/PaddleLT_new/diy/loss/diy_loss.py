#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
diy loss list
"""
import os

if "paddle" in os.environ.get("FRAMEWORK"):
    import paddle

if "torch" in os.environ.get("FRAMEWORK"):
    import torch


def naive_loss_list(logit, loss_list):
    """loss list"""
    for l in loss_list:
        logit = eval(l)
    return logit


def mean_loss(logit):
    """base mean loss"""
    if isinstance(logit, (list, tuple)):
        tmp = 0.0
        count = 0
        for l in logit:
            # NOTE(Aurelius84): Some Tensor calculated from paddle.nonzeros
            # and will return nothing in some cases. It will lead to
            # error when calculating mean.
            if isinstance(l, paddle.Tensor) and l.numel() > 0:
                mean = paddle.mean(l)
                tmp += mean
                count += 1
        # loss = tmp / len(logit)
        loss = tmp / count
        return loss
    elif isinstance(logit, paddle.Tensor):
        loss = paddle.mean(logit)
        return loss
    else:
        raise Exception("something wrong with mean_loss!!")


def torch_mean_loss(logit):
    """torch mean loss"""
    if isinstance(logit, (list, tuple)):
        tmp = 0.0
        count = 0
        for l in logit:
            if isinstance(l, torch.Tensor) and l.numel() > 0:
                mean = torch.mean(l)
                tmp += mean
                count += 1
        # loss = tmp / len(logit)
        loss = tmp / count
        return loss
    elif isinstance(logit, torch.Tensor):
        loss = torch.mean(logit)
        return loss
    else:
        raise Exception("something wrong with torch_mean_loss!!")
