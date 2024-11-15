#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test sin
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestSin(APIBase):
    """
    test sin
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # sin has backward compute
        self.enable_backward = True


obj = TestSin(paddle.sin)


@pytest.mark.api_base_sin_vartype
def test_sin_base():
    """
    base
    when input tensor is float16，cannot support running in cpu mode,it a bug
    """
    x = np.array([-0.3, 5.3, 0.3, -6.2])
    res = np.sin(x)
    obj.base(res=res, x=x)


# def test_sin1():
#     """
#     x=[], returns none in static mode, is a common problem
#     """
#     x = np.array([])
#     res = np.sin(x)
#     obj.run(res=res, x=x)


@pytest.mark.api_base_sin_parameters
def test_sin2():
    """
    x = large num
    """
    x = np.array([-23333, 463333, 665432222])
    res = np.sin(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sin_parameters
def test_sin3():
    """
    x = float
    """
    x = np.array([0.00002, -0.000002, 0.44444])
    res = np.sin(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sin_parameters
def test_sin4():
    """
    x = many dimensions
    """
    x = np.arange(12).reshape(2, 2, 3)
    res = np.sin(x)
    obj.run(res=res, x=x)


@pytest.mark.api_base_sin_parameters
def test_sin5():
    """
    name is defined
    """
    x = np.arange(12).reshape(2, 2, 3)
    res = np.sin(x)
    obj.run(res=res, x=x, name="test_sin")


class TestSin1(APIBase):
    """
    test sin
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.int32]
        self.enable_backward = True


obj1 = TestSin1(paddle.sin)


# @pytest.mark.api_base_sin_exception
# def test_sin6():
#     """
#     TypeError:x=int32(tensor)
#     """
#     x = np.array([3.3, -3.1])
#     obj1.exception(mode="c", etype="NotFoundError", x=x)


@pytest.mark.api_base_sin_exception
def test_sin7():
    """
    TypeError:x=float(no tensor)
    """
    x = 3.3
    obj1.exception(mode="c", etype="InvalidArgumentError", x=x)
