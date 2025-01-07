"""
test mean
"""
from apibase import APIBase, TestWithoutPIR

import paddle
import pytest
import numpy as np


class TestMean(APIBase):
    """
    test paddle.mean api
    """

    def hook(self):
        """
        implement
        """
        self.types = [np.float32, np.float64]
        # self.debug = True
        # self.static = True
        # enable check grad
        self.no_grad_var = ["axis"]
        self.enable_backward = False


obj = TestMean(paddle.mean)


@pytest.mark.api_base_mean_vartype
def test_mean_base():
    """
    base
    """
    x = np.array([[1, 2, 5, 4]])
    res = np.mean(x)
    obj.base(res=res, x=x)


@pytest.mark.api_base_mean_parameters
def test_mean1():
    """
    run axis is int and axis range[-2,2), axis<0
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype("float32")
    res = np.mean(x, axis=-2)
    obj.run(res=res, x=x, axis=-2)


@pytest.mark.api_base_mean_parameters
def test_mean2():
    """
    run axis is int and axis range[-2,2), axis>0
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype("float32")
    res = np.mean(x, axis=1)
    obj.run(res=res, x=x, axis=1)


@pytest.mark.api_base_mean_parameters
def test_mean3():
    """
    run axis is list, axis=[0,1], res should [1, 1, 3],
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 1))
    obj.run(res=res, x=x, axis=[0, 1])


@pytest.mark.api_base_mean_parameters
def test_mean4():
    """
    run axis is list, dim=3, axis=[0,1,2], res should [1, 1, 3],
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 1, 2))
    obj.run(res=res, x=x, axis=[0, 1, 2])


@pytest.mark.api_base_mean_parameters
def test_mean5():
    """
    run axis is int and axis range[-2,2), axis<0, keepdim=False
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype("float64")
    res = np.mean(x, axis=-2, keepdims=False)
    obj.run(res=res, x=x, axis=-2, keepdim=False)


@pytest.mark.api_base_mean_parameters
def test_mean6():
    """
    run axis is int and axis range[-2,2), axis>0, keepdim=True
    :return:
    """
    x = np.array([[1, 2, 5, 4]]).astype("float64")
    res = np.mean(x, axis=1, keepdims=True)
    obj.run(res=res, x=x, axis=1, keepdim=True)


@pytest.mark.api_base_mean_parameters
def test_mean7():
    """
    run axis is list, axis=[0,2], res should [1, 1, 3], keepdim=True
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 2), keepdims=True)
    obj.run(res=res, x=x, axis=[0, 2], keepdim=True)


@pytest.mark.api_base_mean_parameters
def test_mean8():
    """
    run axis is list, dim=3, axis=[0,1,2], res should [1, 1, 3], keepdim=False
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 1, 2), keepdims=False)
    obj.run(res=res, x=x, axis=[0, 1, 2], keepdim=False)


@pytest.mark.api_base_mean_parameters
def test_mean9():
    """
    run axis is tuple, dim=3, axis=(0,2), res should [1, 1, 3], keepdim=True
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 2), keepdims=True)
    obj.run(res=res, x=x, axis=(0, 2), keepdim=True)


@pytest.mark.api_base_mean_parameters
def test_mean10():
    """
    run axis is tuple, dim=3, axis=(0,1,2), res should [1, 1, 3], keepdim=False
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(0, 1, 2), keepdims=False)
    obj.run(res=res, x=x, axis=(0, 1, 2), keepdim=False)


@pytest.mark.api_base_mean_parameters
def test_mean11():
    """
    run axis is tuple, dim=3, axis=(-3,1,2), res should [1, 1, 3], keepdim=False
    static is ok, but dygraph has bug， waitting for fix.
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    res = np.mean(x, axis=(-3, 1, 2), keepdims=False)
    obj.run(res=res, x=x, axis=(0, 1, 2), keepdim=False)


@pytest.mark.api_base_mean_parameters
def test_mean12():
    """
    run axis is int and axis is Tensor, axis<0
    :return:
    """
    paddle.disable_static()
    x = np.array([[1, 2, 5, 4]]).astype("float32")
    res = np.mean(x, axis=-2)
    exp = paddle.mean(paddle.to_tensor(x), axis=paddle.to_tensor(-2))
    assert np.allclose(exp.numpy(), res)


@pytest.mark.api_base_mean_parameters
def test_mean13():
    """
    run axis is Tensor, axis=Tensor([0,1]), res should [1, 1, 3],
    :return:
    """
    x = np.array([[[2.0, 3.0, 4.0]], [[4.0, 3.0, 4.0]]]).astype("float32")
    axis = np.array([0, 1])
    res = np.mean(x, axis=(0, 1))
    with TestWithoutPIR(obj):
        obj.run(res=res, x=x, axis=axis)
