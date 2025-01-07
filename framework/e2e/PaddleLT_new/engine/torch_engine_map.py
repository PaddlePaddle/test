#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
engine 查表
"""

from engine.torch_eval import TorchLayerEval
from engine.torch_train import TorchLayerTrain

# from engine.torch_eval_bm import TorchLayerEvalBM

# from interpreter.testing_trans import TrainTrans, EvalTrans


torch_engine_map = {
    "torch_dy_eval": TorchLayerEval,
    "torch_dy_train": TorchLayerTrain,
    # "torch_dy_eval_perf": TorchLayerEvalBM,  # 动态图评估性能
}
