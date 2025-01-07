#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
nvidia 相关
"""

import GPUtil


def get_nv_memory(device_id):
    """
    字典中key改名
    """
    gpus = GPUtil.getGPUs()
    gpu = gpus[device_id]
    memory_dict = {
        "gpu_name": gpu.name,
        "gpu_id": device_id,
        "Load": f"{gpu.load * 100:.2f}%",
        "Memory_Total": f"{gpu.memoryTotal} MB",
        "Memory_Free": f"{gpu.memoryFree} MB",
        "Memory_Used": f"{gpu.memoryUsed} MB",
    }
    return memory_dict
