#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test kthvalue
"""
from apibase import APIBase
from apibase import randtool
from apibase import compare
import paddle
import pytest
import numpy as np


class TestKthvalue(APIBase):
    """
    test kthvalue
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.static = False
        # self.debug = True
        # enable check grad
        # self.enable_backward = False


obj = TestKthvalue(paddle.kthvalue)
obj1 = TestKthvalue(paddle.kthvalue)
obj1.static = True
obj1.dygraph = False
obj1.enable_backward = False


@pytest.mark.api_base_kthvalue_vartype
def test_kthvalue_base():
    """
    base
    """
    k = 3
    axis = 0
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    res = (np.array([[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]), np.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    obj.base(res=res, x=x, k=k, axis=axis)


@pytest.mark.api_base_kthvalue_vartype
def test_kthvalue_base1():
    """
    base
    """
    k = 3
    axis = 0
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    res = np.array([[17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]])
    obj1.base(res=res, x=x, k=k, axis=axis)


@pytest.mark.api_base_kthvalue_parameters
def test_kthvalue():
    """
    q = 0.75
    axis = 5
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    k = 4
    axis = 2
    keepdim = False
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    res = (np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]]), np.array([[3, 3], [3, 3], [3, 3]]))
    obj.run(res=res, x=x, k=k, axis=axis, keepdim=keepdim)


@pytest.mark.api_base_kthvalue_parameters
def test_kthvalue1():
    """
    q = 0.75
    axis = 3
    keepdims = True
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    k = 4
    axis = 2
    keepdims = True
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    res = (
        np.array([[[4.0], [8.0]], [[12.0], [16.0]], [[20.0], [24.0]]]),
        np.array([[[3], [3]], [[3], [3]], [[3], [3]]]),
    )
    obj.run(res=res, x=x, k=k, axis=axis, keepdim=keepdims)


@pytest.mark.api_base_kthvalue_parameters
def test_kthvalue2():
    """
    q = 0.75
    axis = 5
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    k = 4
    axis = 2
    keepdim = False
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    res = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])
    obj1.run(res=res, x=x, k=k, axis=axis, keepdim=keepdim)


@pytest.mark.api_base_kthvalue_parameters
def test_kthvalue3():
    """
    q = 0.75
    axis = 3
    keepdims = True
    x = randtool("float", -1, 1, (3, 6, 3, 4, 2, 5))
    """
    k = 4
    axis = 2
    keepdims = True
    x = np.arange(1, 25).reshape(3, 2, 4).astype(np.float32)
    if paddle.framework.use_pir_api():
        res = np.array([[[4.0], [8.0]], [[12.0], [16.0]], [[20.0], [24.0]]])
    else:
        # It's a bug for old static program, reslut is same with `keepdims = False` setting.
        res = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])
    obj1.run(res=res, x=x, k=k, axis=axis, keepdim=keepdims)
