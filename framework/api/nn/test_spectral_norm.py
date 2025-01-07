#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test_spectralnorm
"""
import paddle
import pytest
import numpy as np


if paddle.is_compiled_with_cuda():
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    places = [paddle.CPUPlace()]


def cal_spectral_norm(x, dim=0, power_iters=1):
    """
    calculate spectral_norm api
    """
    shape = x.shape
    u = np.array([1] * shape[dim])
    mul = 1
    length = len(shape)
    for i in range(length):
        if i == dim:
            continue
        else:
            mul *= shape[i]

    v = np.array([1] * mul)

    # 重塑为矩阵
    g_dim = list(range(len(shape)))
    g_dim.pop(dim)
    convert_dim = [dim] + g_dim
    # 返回为tensor
    g_dim = list(range(len(shape)))
    back_dim = g_dim[1:]
    back_dim.insert(dim, 0)

    W = np.transpose(x, convert_dim)
    shape1 = W.shape
    W = W.reshape(shape[dim], mul)

    for k in range(power_iters):
        u1 = (np.dot(W, v)) / np.linalg.norm((np.dot(W, v)), ord=2)
        v1 = (np.dot(W.T, u)) / np.linalg.norm((np.dot(W.T, u)), ord=2)
        u, v = u1, v1

    sigma_W = np.linalg.multi_dot([u.T, W, v])
    W = W / (sigma_W + 1e-12)
    W = W.reshape(shape1)
    W = np.transpose(W, back_dim)
    return W


def cal_specal_norm_static(x, place, w0, w1, dim=0, power_iters=1, dtype="float32"):
    """
    calculate static forward
    """
    paddle.enable_static()
    main_program, strartup_program = paddle.static.Program(), paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(main_program=main_program, startup_program=strartup_program):
            data = paddle.static.data(name="data", shape=x.shape, dtype=dtype)
            spectral_norm = paddle.nn.SpectralNorm(x.shape, dim, power_iters=power_iters, dtype=dtype)
            output = spectral_norm(data)
            exe = paddle.static.Executor(place=place)
            exe.run(strartup_program)
            res = exe.run(
                main_program,
                feed={"data": x, "spectral_norm_0.w_0": w0, "spectral_norm_0.w_1": w1},
                fetch_list=[output],
            )
    return res[0]


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm0():
    """
    default
    x: 2-D tensor
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(2, 4).astype("float32")
        expect = cal_spectral_norm(x_data)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1, 1]).astype(np.float32)
        weight_v = np.array([1] * 4).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 0, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm1():
    """
    default
    x: 3-D tensor
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(2, 4, 5).astype("float32")
        expect = cal_spectral_norm(x_data)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1, 1]).astype(np.float32)
        weight_v = np.array([1] * 20).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 0, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            # Don't test feed the parameter in pir.
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm2():
    """
    default
    x: 4-D tensor
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(2, 4, 5, 6).astype("float32")
        expect = cal_spectral_norm(x_data)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1, 1]).astype(np.float32)
        weight_v = np.array([1] * 120).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 0, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm3():
    """
    default
    x: 4-D tensor
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(4, 3, 5, 6).astype("float32")
        expect = cal_spectral_norm(x_data)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1] * 4).astype(np.float32)
        weight_v = np.array([1] * 90).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 0, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm4():
    """
    default
    x: 5-D tensor
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(4, 3, 2, 5, 6).astype("float32")
        expect = cal_spectral_norm(x_data)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1] * 4).astype(np.float32)
        weight_v = np.array([1] * 180).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 0, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm5():
    """
    x: 5-D tensor
    dim = 1
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(4, 3, 2, 5, 6).astype("float32")
        expect = cal_spectral_norm(x_data, dim=1)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1] * 3).astype(np.float32)
        weight_v = np.array([1] * 240).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 1, power_iters=1)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v, dim=1)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm6():
    """
    x: 5-D tensor
    dim = 1
    power_iters = 10
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(4, 3, 5).astype("float32") * 40
        expect = cal_spectral_norm(x_data, dim=1, power_iters=10)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1] * 3).astype(np.float32)
        weight_v = np.array([1] * 20).astype(np.float32)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 1, power_iters=10)
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(x_data, place, w0=weight_u, w1=weight_v, dim=1, power_iters=10)
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)


@pytest.mark.api_nn_SpectralNorm_parameters
def test_spectralnorm7():
    """
    x: 5-D tensor
    dim = 1
    power_iters = 10
    dtype = 'float64'
    """
    for place in places:
        paddle.disable_static(place=place)
        np.random.seed(22)
        x_data = np.random.rand(4, 3, 5).astype("float64") * 40
        expect = cal_spectral_norm(x_data, dim=1, power_iters=10)
        x = paddle.to_tensor(x_data)
        weight_u = np.array([1] * 3).astype(np.float64)
        weight_v = np.array([1] * 20).astype(np.float64)
        spectral_norm = paddle.nn.SpectralNorm(x.shape, 1, power_iters=10, dtype="float64")
        spectral_norm._parameters["weight_u"].set_value(paddle.to_tensor(weight_u))
        spectral_norm._parameters["weight_v"].set_value(paddle.to_tensor(weight_v))
        spectral_norm_out = spectral_norm(x)

        if not paddle.framework.use_pir_api():
            static_res = cal_specal_norm_static(
                x_data, place, w0=weight_u, w1=weight_v, dim=1, power_iters=10, dtype="float64"
            )
            assert np.allclose(static_res, spectral_norm_out.numpy())
        assert np.allclose(spectral_norm_out.numpy(), expect, atol=1e-4)
