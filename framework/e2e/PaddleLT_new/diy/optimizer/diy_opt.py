#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
diy optimizer
"""
import os

if "paddle" in os.environ.get("FRAMEWORK"):
    import paddle

if "torch" in os.environ.get("FRAMEWORK"):
    import torch


def naive_opt(net, opt_api, learning_rate):
    """navie optimizer func"""
    opt = eval(opt_api)(learning_rate=learning_rate, parameters=net.parameters())
    return opt


def torch_opt(net, opt_api, learning_rate):
    """torch optimizer func"""
    opt = eval(opt_api)(net.parameters(), lr=learning_rate)
    return opt
