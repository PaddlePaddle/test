#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test linspace
"""
from apibase import APIBase
import paddle
import pytest
import numpy as np


class TestLinspace(APIBase):
    """
    test linspace
    """

    def hook(self):
        """
        implement
        """
        # start and stop only support float32, float64 tensor
        self.types = [np.float32, np.float64]
        self.enable_backward = False
        self.debug = True


obj = TestLinspace(paddle.linspace)


@pytest.mark.api_base_linspace_vartype
def test_linspace_base():
    """
    base
    """
    start = 3.5
    stop = 9.6
    num = 5
    res = np.linspace(start=start, stop=stop, num=num)
    obj.base(res=res, start=start, stop=stop, num=num)


@pytest.mark.api_base_linspace_parameters
def test_linspace1():
    """
    start=tensor,stop=num
    """
    start = 2.9 + np.arange(1)
    stop = 9.6
    num = 5
    res = np.array([2.9, 4.575, 6.25, 7.925, 9.6])
    obj.run(res=res, start=start, stop=stop, num=num, dtype="float64")


@pytest.mark.api_base_linspace_parameters
def test_linspace2():
    """
    start=tensor,stop=tensor
    """
    start = 3.0 + np.arange(1)
    stop = 8.3 + np.arange(1)
    num = 2
    res = np.linspace(start=3.0, stop=8.3, num=num)
    obj.run(res=res, start=start, stop=stop, num=num, dtype="float64")


@pytest.mark.api_base_linspace_parameters
def test_linspace4():
    """
    num=1
    """
    start = 3
    stop = 9
    num = 1
    res = np.linspace(start=start, stop=stop, num=num)
    obj.run(res=res, start=start, stop=stop, num=num)


@pytest.mark.api_base_linspace_exception
def test_linspace5():
    """
    TypeError:num=float
    Cannot catch exception(it's a bug)
    """
    start = 3
    stop = 9
    num = 1.2
    obj.exception(mode="python", etype=TypeError, star=start, stop=stop, num=num)


@pytest.mark.api_base_linspace_parameters
def test_linspace6():
    """
    start>stop
    """
    start = 9
    stop = 3
    num = 5
    res = np.linspace(start=start, stop=stop, num=num)
    obj.run(res=res, start=start, stop=stop, num=num)


@pytest.mark.api_base_linspace_parameters
def test_linspace7():
    """
    dtype=np.dtype
    """
    start = 3
    stop = 9
    num = 5
    dtype = np.float64
    res = np.linspace(start=start, stop=stop, num=num)
    obj.run(res=res, start=start, stop=stop, num=num, dtype=dtype)


@pytest.mark.api_base_linspace_parameters
def test_linspace8():
    """
    dtype=float32
    """
    start = 3
    stop = 9
    num = 5
    dtype = "float32"
    res = np.linspace(start=start, stop=stop, num=num)
    obj.run(res=res, start=start, stop=stop, num=num, dtype=dtype)


@pytest.mark.api_base_linspace_parameters
def test_linspace9():
    """
    dtype=float64
    """
    start = 3
    stop = 9
    num = 5
    dtype = "float64"
    res = np.linspace(start=start, stop=stop, num=num)
    obj.run(res=res, start=start, stop=stop, num=num, dtype=dtype)


@pytest.mark.api_base_linspace_exception
def test_linspace10():
    """
    TypeError:dtype=bool
    """
    start = 3
    stop = 9
    num = 5
    dtype = "bool"
    obj.exception(mode="c", etype="NotFoundError", start=start, stop=stop, num=num, dtype=dtype)


class TestLinspace1(APIBase):
    """
    test linspace
    """

    def hook(self):
        """
        implement
        """
        # num only support int32 tensor
        self.types = [np.int32]
        self.enable_backward = False


obj1 = TestLinspace1(paddle.linspace)


@pytest.mark.api_base_linspace_parameters
def test_linspace11():
    """
    num=tensor
    """
    start = 4.5
    stop = 8
    num = 2 + np.arange(1)
    res = np.linspace(start=start, stop=stop, num=2)
    obj1.run(res=res, start=start, stop=stop, num=num)


@pytest.mark.api_base_linspace_parameters
def test_linspace12():
    """
    input and stop is int32 tensor
    """
    start = 1 + np.arange(1)
    stop = 8 + np.arange(1)
    num = 2
    res = np.array([1, 8])
    obj1.run(res=res, start=start, stop=stop, num=num)
