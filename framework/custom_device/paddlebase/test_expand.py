#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
test_expand.py
"""

from apibase import APIBase
from apibase import randtool
from apibase import compare

import paddle
import pytest
import numpy as np


class TestExpand(APIBase):
    """
    test expand
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # enable check grad
        self.no_grad_var = {"x", "shape"}
        # self.enable_backward = False


obj = TestExpand(paddle.expand)


@pytest.mark.api_base_expand_vartype
def test_expand_base():
    """
    base,shape=list
    """
    x = np.array([1.1, 2, 3])
    shape = [3, 3]
    res = np.array([[1.1, 2, 3], [1.1, 2, 3], [1.1, 2, 3]])
    obj.base(res=res, x=x, shape=shape)


obj.enable_backward = False


@pytest.mark.api_base_expand_parameters
def test_expand1():
    """
    test shape = tuple
    shape = (2, 3),x_type=np.int32
    """
    x = np.array([1, 2, 3]).astype(np.int32)
    shape = (3, 3)
    res = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand2():
    """
    shape = (3,2),x_type=np.int64
    """
    x = np.array([[1], [2], [3]]).astype(np.int64)
    shape = (3, 2)
    res = np.array([[1, 1], [2, 2], [3, 3]])
    obj.run(res=res, x=x, shape=shape)


# test shape = list
# def test_expand3():
#     """
#     shape = [1],x_type=np.bool_,todo:fix RC
#     """
#     x = np.array([False, True]).astype(np.bool_)
#     shape = [1, 2]
#     res = np.array([[False, True]])
#     obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand4():
    """
    shape_size=x_shape
    """
    x = np.array([1]).astype("int32")
    shape = [1]
    res = x
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand5():
    """
    shape = list,x_type='int64'
    """
    x = np.array([1, 2, 3]).astype("int64")
    shape = [2, 3]
    res = np.array([[1, 2, 3], [1, 2, 3]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand6():
    """
    shape = [1,-1]
    """
    x = np.array([1, 2]).astype("int64")
    shape = [1, -1]
    res = np.array([[1, 2]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand8():
    """
    shape.shape = 6
    """
    x = np.array([1]).astype(np.int64)
    shape = [1, 1, 1, 1, 1, 1]
    res = np.array([[[[[[1]]]]]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand9():
    """
    shape = [1],x_shape=shape_size
    """
    x = np.array([1])
    shape = [1]
    res = np.array([1])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand10():
    """
    shape = [2,]
    """
    x = np.array([1])
    shape = [2]
    res = np.array([1, 1])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand11():
    """
    shape = np(Tensor)
    shape = np.array([1])
    """
    x = np.array([1.0])
    shape = np.array([1, 1]).astype(np.int32)
    res = np.array([[1.0]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand12():
    """
    shape = 2-D
    """
    x = np.array([[1, 2], [2, 3]])
    shape = np.array([1, 2, 2]).astype(np.int32)
    res = np.array([[[1, 2], [2, 3]]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand13():
    """
    shape = np.array([6])
    """
    x = np.array([[[1]]])
    shape = np.array([2, 3, 1]).astype(np.int32)
    res = np.array([[[1], [1], [1]], [[1], [1], [1]]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand14():
    """
    shape_value = -1
    """
    x = np.array([1])
    shape = np.array([6, -1]).astype(np.int32)
    res = np.array([[1], [1], [1], [1], [1], [1]])
    obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_exception
def test_expand15():
    """
    shape.shape >6,c++error
    """
    x = np.array([1])
    shape = np.ones([7]).astype(np.int32)
    obj.exception(x=x, shape=shape, mode="c", etype="InvalidArgument")


# @pytest.mark.api_base_expand_exception
# def test_expand16():
#     """
#     x_shape < 0,c++error,static AssertionError
#     """
#     x = np.array([[1], [1]])
#     shape = [2, 0]
#     obj.exception(x=x, shape=shape, mode="c", etype="InvalidArgument")


@pytest.mark.api_base_expand_exception
def test_expand17():
    """
    shape = 2-D,static,TypeError
    """
    x = np.array([1, 2]).astype(np.int64)
    shape = [[1], [2], [3]]
    obj.exception(x=x, shape=shape, mode="c", etype="InvalidType")


# def test_expand18():
#     """
#     shape = list,x_type='float16'
#     """
#     x = np.array([1, 2, 3]).astype('float16')
#     shape = [2, 3]
#     res = np.array([[1, 2, 3], [1, 2, 3]])
#     obj.run(res=res, x=x, shape=shape)


@pytest.mark.api_base_expand_parameters
def test_expand19():
    """
    shape = [1, 3, 4, 4, 1, 1]
    """
    paddle.disable_static()
    x = paddle.ones([1, 3, 1, 1, 1, 1])
    x.stop_gradient = False
    y = paddle.expand(x, [1, 3, 4, 4, 1, 1])
    y = y.sum()
    y.backward(retain_graph=True)
    res = x.grad
    exp = np.array([[[[[[16.0]]]], [[[[16.0]]]], [[[[16.0]]]]]])

    compare(res.numpy(), exp)
