#!/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tool
"""

import logging
import numpy as np
import paddle


def _randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)
    elif dtype == "int32":
        return np.random.randint(low, high, shape).astype("int32")
    elif dtype == "int64":
        return np.random.randint(low, high, shape).astype("int64")
    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)
    elif dtype == "float16":
        return low + (high - low) * np.random.random(shape).astype("float16")
    elif dtype == "float32":
        return low + (high - low) * np.random.random(shape).astype("float32")
    elif dtype == "float64":
        return low + (high - low) * np.random.random(shape).astype("float64")
    elif dtype in ["complex", "complex64", "complex128"]:
        data = low + (high - low) * np.random.random(shape) + (low + (high - low) * np.random.random(shape)) * 1j
        return data if dtype == "complex" or "complex128" else data.astype(np.complex64)
    elif dtype == "bool":
        data = np.random.randint(0, 2, shape).astype("bool")
        return data
    else:
        assert False, "dtype is not supported"


# def compare(result, expect, delta=1e-10, rtol=1e-10):
#     """
#     比较函数
#     :param result: 输入值
#     :param expect: 输出值
#     :param delta: 误差值
#     :param rtol: 相对误差
#     :return:
#     """
#     if isinstance(expect, paddle.Tensor) or isinstance(expect, np.ndarray):
#         if isinstance(result, paddle.Tensor):
#             result = result.numpy()
#         if isinstance(expect, paddle.Tensor):
#             expect = expect.numpy()
#         res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
#         # 出错打印错误数据
#         if res is False:
#             diff = abs(result - expect)
#             logging.error("expect is: {}".format(expect))
#             logging.error("result is: {}".format(result))
#             logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
#         if result.dtype != expect.dtype:
#             logging.error(
#                 "Different output data types! res type is: {}, and expect type is: {}".format(
#                     result.dtype, expect.dtype
#                 )
#             )
#         assert res
#         assert result.shape == expect.shape
#         assert result.dtype == expect.dtype
#     elif isinstance(expect, list) or isinstance(expect, tuple):
#         for i, element in enumerate(expect):
#             if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, paddle.Tensor):
#                 if i > 0:
#                     break
#                 compare(result, expect[i], delta, rtol)
#
#             else:
#                 compare(result[i], expect[i], delta, rtol)
#     elif isinstance(expect, (bool, int, float)):
#         assert expect == result
#     elif expect is None:
#         pass
#     else:
#         raise Exception("expect is unknown data struction in compare_tool!!!")
