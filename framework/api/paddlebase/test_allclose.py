#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test allclose
"""
from apibase import APIBase, TestWithoutPIR
import paddle
import pytest
import numpy as np


class TestAllclose(APIBase):
    """
    test allclose
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        self.enable_backward = False


class TestAllclose1(APIBase):
    """
    test allclose
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float64]
        self.enable_backward = False


obj = TestAllclose(paddle.allclose)
obj2 = TestAllclose1(paddle.allclose)


@pytest.mark.api_base_allclose_vartype
def test_allclose_base():
    """
    base
    return True
    """
    x = np.array([10.00001])
    y = np.array([10])
    a = 0.01
    r = 0.01
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=False)
    res = np.array(res)
    obj.base(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=False)


@pytest.mark.api_base_allclose_parameters
def test_allclose1():
    """
    paddle.allclose() returns False when the difference equals threshold,it's a bug
    """
    x = np.array([10.1])
    y = np.array([10])
    a = 0.0
    r = 0.01
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=False)
    res = np.array(res)
    with TestWithoutPIR(obj2):
        obj2.run(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=False)


@pytest.mark.api_base_allclose_parameters
def test_allclose2():
    """
    return false
    """
    x = np.array([10.1])
    y = np.array([10])
    a = 0.001
    r = 0.000001
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=False)
    res = np.array(res)
    obj.run(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=False)


@pytest.mark.api_base_allclose_parameters
def test_allclose3():
    """
    equal_nan=True and doesn't exist nan
    """
    x = np.array([10.1, 0.001, 4000000])
    y = np.array([10, 0.001, 4000000])
    a = 0.001
    r = 0.000001
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=True)
    res = np.array(res)
    obj.run(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=True)


@pytest.mark.api_base_allclose_parameters
def test_allclose4():
    """
    equal_nan=True and exist nan
    """
    x = np.array([np.nan])
    y = np.array([np.nan])
    a = 0.001
    r = 0.00001
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=True)
    res = np.array(res)
    obj.run(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=True)


@pytest.mark.api_base_allclose_parameters
def test_allclose5():
    """
    equal_nan=False and exist nan
    """
    x = np.array([np.nan])
    y = np.array([np.nan])
    a = 0.001
    r = 0.00001
    res = np.allclose(x, y, rtol=r, atol=a, equal_nan=False)
    res = np.array(res)
    obj.run(res=res, x=x, y=y, rtol=r, atol=a, equal_nan=False)


@pytest.mark.api_base_allclose_parameters
def test_allclose6():
    """
    use default value
    """
    x = np.array([10.001])
    y = np.array([10.00001])
    res = np.allclose(x, y)
    res = np.array(res)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_allclose_parameters
def test_allclose7():
    """
    input many dimensions
    """
    x = 0.1 + np.arange(24).reshape(2, 2, 2, 3)
    y = np.arange(24).reshape(2, 2, 2, 3)
    res = np.allclose(x, y)
    res = np.array(res)
    obj.run(res=res, x=x, y=y)


@pytest.mark.api_base_allclose_parameters
def test_allclose8():
    """
    rtol=negtive num && atol=negtive num
    """
    x = 0.1 + np.arange(24).reshape(2, 2, 2, 3)
    y = np.arange(24).reshape(2, 2, 2, 3)
    a = -2.0
    r = -3.0
    res = np.allclose(x, y, rtol=r, atol=a)
    res = np.array(res)
    obj.run(res=res, x=x, y=y, rtol=r, atol=a)


@pytest.mark.api_base_allclose_parameters
def test_allclose9():
    """
    x and y is []
    """
    x = np.array([])
    y = np.array([])
    res = np.allclose(x, y)
    res = np.array(res)
    obj.run(res=res, x=x, y=y)


class TestAllclose1(APIBase):
    """
    test allclose
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float16]
        self.debug = True
        self.enable_backward = False


obj1 = TestAllclose1(paddle.allclose)


@pytest.mark.api_base_allclose_exception
def test_allclose10():
    """
    x and y is float16
    """
    x = np.array([10.001])
    y = np.array([10.00001])
    obj1.exception(mode="c", etype="NotFound", x=x, y=y)
