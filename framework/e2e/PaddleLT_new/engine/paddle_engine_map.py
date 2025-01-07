#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
engine 查表
"""

from engine.paddle_train import LayerTrain
from engine.paddle_eval import LayerEval
from engine.paddle_export import LayerExport
from engine.paddle_infer import LayerInfer
from engine.paddle_eval_bm import LayerEvalBM
from engine.paddle_train_bm import LayerTrainBM


paddle_engine_map = {
    "dy_train": LayerTrain,  # 动态图训练
    "dy_dp_train": LayerTrain,
    "dy2st_train": LayerTrain,  # 动转静训练
    "dy2st_train_inputspec": LayerTrain,  # 动转静训练+动态inputspec
    "dy2st_train_static_inputspec": LayerTrain,  # 动转静训练+静态inputspec
    "dy2st_train_cinn": LayerTrain,  # CINN训练
    "dy2st_train_cinn_inputspec": LayerTrain,  # CINN训练+动态inputspec
    "dy2st_train_cinn_static_inputspec": LayerTrain,  # CINN训练+静态inputspec
    "dy_eval": LayerEval,  # 动态图评估
    "dy2st_eval": LayerEval,  # 动转静评估
    "dy2st_eval_inputspec": LayerEval,  # 动转静评估+动态inputspec
    "dy2st_eval_static_inputspec": LayerEval,  # 动转静评估+静态inputspec
    "dy2st_eval_cinn": LayerEval,  # CINN评估
    "dy2st_eval_cinn_inputspec": LayerEval,  # CINN评估+动态inputspec
    "dy2st_eval_cinn_static_inputspec": LayerEval,  # CINN评估+静态inputspec
    "jit_save": LayerExport,  # 动转静导出
    "jit_save_inputspec": LayerExport,  # 动转静+动态inputspec导出
    "jit_save_static_inputspec": LayerExport,  # 动转静+静态inputspec导出
    "jit_save_cinn": LayerExport,  # 动转静CINN导出
    "jit_save_cinn_inputspec": LayerExport,  # 动转静CINN+动态inputspec导出
    "jit_save_cinn_static_inputspec": LayerExport,  # 动转静CINN+静态inputspec导出
    "paddle_infer_gpu": LayerInfer,  # gpu预测
    "paddle_infer_cpu": LayerInfer,  # cpu预测
    "paddle_infer_mkldnn": LayerInfer,  # cpu mkldnn预测
    "paddle_infer_ort": LayerInfer,  # ort预测
    "paddle_infer_new_exc_pir": LayerInfer,  # 新执行器+PIR预测
    "dy_eval_perf": LayerEvalBM,  # 动态图评估性能
    "dy2st_eval_perf": LayerEvalBM,  # 动转静评估性能
    "dy2st_eval_cinn_perf": LayerEvalBM,  # CINN评估性能
    "dy2st_eval_cinn_perf_pre": LayerEvalBM,  # pre测试
    "dy_train_perf": LayerTrainBM,  # 动态图评估性能
    "dy2st_train_perf": LayerTrainBM,  # 动转静评估性能
    "dy2st_train_cinn_perf": LayerTrainBM,  # CINN评估性能
    "dy2st_train_cinn_perf_pre": LayerTrainBM,  # pre测试
}
