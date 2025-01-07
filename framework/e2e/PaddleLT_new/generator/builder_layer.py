#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer builder
"""

import os

# if os.environ.get("FRAMEWORK") == "paddle":
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

        os.system("cd /root/.paddlenlp && rm -rf models")

if "torch" in os.environ.get("FRAMEWORK"):
    import torch
    import torch_case


class BuildLayer(object):
    """BuildLayer"""

    def __init__(self, layerfile):
        """init"""
        self.layername = layerfile + ".LayerCase"

    def get_layer(self):
        """get_layer"""
        layer = eval(self.layername)()
        return layer
