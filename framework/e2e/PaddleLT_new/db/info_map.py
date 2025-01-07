#!/bin/env python
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
信息map
"""

precision_md5 = {
    "paddlelt_eval_cinn": "e4929c3e2b9fb8c48ab11f8e727bb82f",
    "paddlelt_train_cinn": "f4e5663420400caf5912b8288fb9c58c",
    "paddlelt_eval_cinn_inputspec": "2b4ab87e8f4a24f3f7087a2c2be2b055",
    "paddlelt_train_cinn_inputspec": "0b00a671d6e8db2e16afc619c7289970",
    "paddlelt_train_api_dy2stcinn_static_inputspec": "7eeb0c154aaad256782c7972c1d5dde4",
    "paddlelt_train_api_dy2stcinn_inputspec": "76017cfeb6074f7188253df556e9fef9",
    "paddlelt_train_prim_inputspec": "33f3b8b4505041abe5ae221f2abd8932",
    "paddlelt_train_pir_infersymbolic_inputspec": "07da2ef04135d7ec5d42987705204e1f",
    "paddlelt_train_nlp_dy2stcinn_inputspec": "65e6644742b2653427cdaf51035e5ef6",
    "paddlelt_train_ocr_dy2stcinn_inputspec": "d3cd6b167556057f25568d93e0592529",
}

precision_flags = {
    "paddlelt_eval_cinn": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_cinn": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_eval_cinn_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_cinn_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_api_dy2stcinn_static_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_api_dy2stcinn_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_prim_inputspec": {
        "FLAGS_prim_all": "true",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_enable_pir_in_executor": "1",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_pir_infersymbolic_inputspec": {
        "FLAGS_prim_all": "true",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_enable_pir_api": "1",
        "FLAGS_enable_pir_in_executor": "1",
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_check_infer_symbolic": "1",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_nlp_dy2stcinn_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
    "paddlelt_train_ocr_dy2stcinn_inputspec": {
        "MIN_GRAPH_SIZE": "0",
        "FLAGS_prim_all": "true",
        "FLAGS_use_cinn": "1",
        "FLAGS_prim_enable_dynamic": "true",
        "FLAGS_prim_forward_blacklist": "pd_op.dropout",
    },
}

precision_ci_md5 = {
    "CI_paddlelt_train_cinn_eval_cinn_inputspec": "4ed830834040691d336abc8d4bb9f7f7",
}

performance_md5 = {
    "paddlelt_perf_1000_cinn_cinn_A100_latest_as_baseline": "1f7253c6a9014bacc74d07cfd3b14ab2",
    "paddlelt_train_perf_1000_cinn_cinn_A100_latest_as_baseline": "1f7253c6a9014bacc74d07cfd3b14ab2",
    "paddlelt_perf_1000kernel_cinn_cinn_A100": "1f7253c6a9014bacc74d07cfd3b14ab2",
    "paddlelt_train_perf_1000kernel_cinn_cinn_A100": "1f7253c6a9014bacc74d07cfd3b14ab2",
}
