#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_is_grad_enabled
"""


import paddle
import pytest


@pytest.mark.api_nn_Orthogonal_parameters
def test_is_grad_enabled0():
    """
    test01
    """
    r1 = paddle.is_grad_enabled()
    assert r1


@pytest.mark.api_nn_Orthogonal_parameters
def test_is_grad_enabled1():
    """
    test02
    """
    with paddle.set_grad_enabled(False):
        r2 = paddle.is_grad_enabled()  # False
        assert not r2


@pytest.mark.api_nn_Orthogonal_parameters
def test_is_grad_enabled2():
    """
    test03
    """
    paddle.enable_static()
    r3 = paddle.is_grad_enabled()
    assert r3
