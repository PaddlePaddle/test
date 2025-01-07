#!/bin/env python3
# -*- coding: utf-8 -*-
# @author Zeref996
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
runner
"""

import os
import json

# import logging
import traceback
import numpy as np

from pltools.logger import Logger

# framework = ""
if "paddle" in os.environ.get("FRAMEWORK"):
    import paddle

    # framework = "paddle"
if "torch" in os.environ.get("FRAMEWORK"):
    import torch

    # framework = "torch"


def base_compare(result, expect, res_name, exp_name, logger, delta=1e-10, rtol=1e-10, exc_dict={}):
    """
    比较函数
    :param result: 待测值
    :param expect: 基线值
    :param delta: 误差值
    :param rtol: 相对误差
    :return:
    """
    if isinstance(result, str):
        raise Exception("result is exception !!!")
    if isinstance(expect, str):
        raise Exception("expect is exception !!!")

    if expect is None or result is None:
        if expect is None:
            Logger("PLT_compare").get_log().info(f"{exp_name} 结果为None, 所以跳过 {exp_name} 和 {res_name} 精度对比")
        if result is None:
            Logger("PLT_compare").get_log().info(f"{res_name} 结果为None, 所以跳过 {exp_name} 和 {res_name} 精度对比")
        pass
    elif isinstance(expect, eval("paddle.Tensor")) or isinstance(expect, np.ndarray):
        if isinstance(result, eval("paddle.Tensor")):
            result = result.numpy()
        if isinstance(expect, eval("paddle.Tensor")):
            expect = expect.numpy()
        # res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # # 出错打印错误数据
        # if res is False:
        #     diff = abs(result - expect)
        #     # logger.error("{} is: {}".format(exp_name, expect))
        #     # logger.error("{} is: {}".format(res_name, result))
        #     logger.error("{} and {} has diff! max diff: {}".format(exp_name, res_name, np.amax(diff)))

        try:
            np.testing.assert_allclose(actual=result, desired=expect, atol=delta, rtol=rtol, equal_nan=True)

            if result.dtype != expect.dtype:
                logger.warn(
                    "Different output data types! res type is: {}, and expect type is: {}".format(
                        result.dtype, expect.dtype
                    )
                )
            # assert res
            assert result.shape == expect.shape
            assert result.dtype == expect.dtype
        except Exception:
            exc_dict[res_name] = traceback.format_exc()
            logger.warn(traceback.format_exc())

    elif isinstance(expect, dict):
        if "multi_result" in result:
            # 专用于多个结果比较, 例如多种inputspec. 只有result会有多个结果, 想法expect固定为一个
            for i, logit_dict in enumerate(result["multi_result"]):
                base_compare(
                    result=logit_dict,
                    expect=expect,
                    res_name=res_name + f"multi_result[{i}]",
                    exp_name=exp_name,
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
        else:
            for k, v in expect.items():
                if k in result:
                    base_compare(
                        result=result[k],
                        expect=expect[k],
                        res_name=res_name + "[{}]".format(str(k)),
                        exp_name=exp_name + "[{}]".format(str(k)),
                        logger=logger,
                        delta=delta,
                        rtol=rtol,
                        exc_dict=exc_dict,
                    )
                else:
                    Logger("PLT_compare").get_log().info(f"{exp_name} 有 {k}, 但是 {res_name} 没有 {k}, 所以跳过 {k} 精度对比")
    elif isinstance(expect, list) or isinstance(expect, tuple):
        for i, element in enumerate(expect):
            if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, eval("paddle.Tensor")):
                if i > 0:
                    break
                base_compare(
                    result=result,
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
            else:
                base_compare(
                    result=result[i],
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
    elif isinstance(expect, (bool, int, float)):
        assert expect == result
    else:
        raise Exception("expect is unknown data struction in compare_tool!!!")

    return exc_dict


def infer_compare(result, expect, res_name, exp_name, logger, delta=1e-10, rtol=1e-10, exc_dict={}):
    """
    比较函数
    :param result: 待测值
    :param expect: 基线值
    :param delta: 误差值
    :param rtol: 相对误差
    :return:
    """
    # 去除反向结果的数据
    forward_handled_result = {"logit": []}
    forward_handled_expect = {"logit": []}

    # 去除非tensor数值的影响
    if isinstance(expect["logit"], (tuple, list)):
        for item in expect["logit"]:
            if not isinstance(item, (int, bool, float)):
                forward_handled_expect["logit"].append(item)

    if isinstance(result["logit"], (tuple, list)):
        for item in result["logit"]:
            if not isinstance(item, (int, bool, float)):
                forward_handled_result["logit"].append(item)

    exc_dict = base_compare(
        result=forward_handled_result,
        expect=forward_handled_expect,
        res_name=res_name,
        exp_name=exp_name,
        logger=logger,
        delta=delta,
        rtol=rtol,
        exc_dict=exc_dict,
    )
    return exc_dict


def torch_compare(result, expect, res_name, exp_name, logger, delta=1e-10, rtol=1e-10, exc_dict={}):
    """
    比较函数
    :param result: 待测值
    :param expect: 基线值
    :param delta: 误差值
    :param rtol: 相对误差
    :return:
    """
    if isinstance(result, str):
        raise Exception("result is exception !!!")
    if isinstance(expect, str):
        raise Exception("expect is exception !!!")

    if expect is None or result is None:
        if expect is None:
            Logger("PLT_compare").get_log().info(f"{exp_name} 结果为None, 所以跳过 {exp_name} 和 {res_name} 精度对比")
        if result is None:
            Logger("PLT_compare").get_log().info(f"{res_name} 结果为None, 所以跳过 {exp_name} 和 {res_name} 精度对比")
        pass
    elif torch.is_tensor(expect) or isinstance(expect, np.ndarray):
        if isinstance(result, eval("paddle.Tensor")):
            result = result.numpy()
        if torch.is_tensor(expect):
            # expect = expect.numpy()
            expect = expect.cpu()
            expect = expect.detach().numpy()
        # res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # # 出错打印错误数据
        # if res is False:
        #     diff = abs(result - expect)
        #     # logger.error("{} is: {}".format(exp_name, expect))
        #     # logger.error("{} is: {}".format(res_name, result))
        #     logger.error("{} and {} has diff! max diff: {}".format(exp_name, res_name, np.amax(diff)))

        try:
            np.testing.assert_allclose(actual=result, desired=expect, atol=delta, rtol=rtol, equal_nan=True)

            if result.dtype != expect.dtype:
                logger.warn(
                    "Different output data types! res type is: {}, and expect type is: {}".format(
                        result.dtype, expect.dtype
                    )
                )
            # assert res
            assert result.shape == expect.shape
            assert result.dtype == expect.dtype
        except Exception:
            exc_dict[res_name] = traceback.format_exc()
            logger.warn(traceback.format_exc())

    elif isinstance(expect, dict):
        if "multi_result" in result:
            # 专用于多个结果比较, 例如多种inputspec. 只有result会有多个结果, 想法expect固定为一个
            for i, logit_dict in enumerate(result["multi_result"]):
                torch_compare(
                    result=logit_dict,
                    expect=expect,
                    res_name=res_name + f"multi_result[{i}]",
                    exp_name=exp_name,
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
        else:
            for k, v in expect.items():
                if k in result:
                    torch_compare(
                        result=result[k],
                        expect=expect[k],
                        res_name=res_name + "[{}]".format(str(k)),
                        exp_name=exp_name + "[{}]".format(str(k)),
                        logger=logger,
                        delta=delta,
                        rtol=rtol,
                        exc_dict=exc_dict,
                    )
                else:
                    Logger("PLT_compare").get_log().info(f"{exp_name} 有 {k}, 但是 {res_name} 没有 {k}, 所以跳过 {k} 精度对比")
    elif isinstance(expect, list) or isinstance(expect, tuple):
        for i, element in enumerate(expect):
            if isinstance(result, (np.generic, np.ndarray)) or isinstance(result, eval("paddle.Tensor")):
                if i > 0:
                    break
                torch_compare(
                    result=result,
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
            else:
                torch_compare(
                    result=result[i],
                    expect=expect[i],
                    res_name=res_name + "[{}]".format(str(i)),
                    exp_name=exp_name + "[{}]".format(str(i)),
                    logger=logger,
                    delta=delta,
                    rtol=rtol,
                    exc_dict=exc_dict,
                )
    elif isinstance(expect, (bool, int, float)):
        assert expect == result
    else:
        raise Exception("expect is unknown data struction in compare_tool!!!")

    return exc_dict


def perf_compare_legacy(baseline, latest):
    """
    比较函数
    :param latest: 待测值
    :param baseline: 基线值
    :return: 比例值
    """
    if isinstance(baseline, str) or isinstance(baseline, str):
        res = "error"
    else:
        if baseline == 0 or latest == 0:
            res = 0
        else:
            if latest > baseline:
                res = (latest / baseline) * -1
            else:
                res = baseline / latest
    return res


def perf_compare(baseline, latest):
    """
    比较函数
    :param latest: 待测值
    :param baseline: 基线值
    :return: 比例值
    """
    if isinstance(baseline, str) or isinstance(baseline, str):
        res = "error"
        return res
    else:
        if baseline == 0 or latest == 0:
            res = 0
        else:
            if latest > baseline:
                res = (latest - baseline) / baseline * -1
            else:
                res = (baseline - latest) / latest
    return "{:.2f}%".format(res * 100)


def perf_compare_dict_legacy(baseline_dict, data_dict, error_list, baseline_layer_type, latest_layer_type):
    """
    生成对比dict
    :param data_dict: 待测字典
    :param baseline_dict: 基线字典
    :param error_list: list[报错子图case]
    :param baseline_layer_type: 基线子图种类，例如layercase、layerApicase
    :param latest_layer_type: 待测子图种类，例如layercase、layerApicase
    :return: 比较字典
    """
    compare_dict = {}
    for title, perf_dict in data_dict.items():
        if title not in error_list:
            layer_case = title.split("^", 1)[1]
            baseline_title = "^".join([baseline_layer_type, layer_case])
            layer_title = "^".join([latest_layer_type, layer_case])
            if baseline_title in baseline_dict and layer_title in data_dict:
                compare_dict[layer_case] = {}
                for perf_engine, t in perf_dict.items():
                    compare_dict[layer_case][perf_engine + "^" + latest_layer_type + "^latest"] = t
                    compare_dict[layer_case][perf_engine + "^" + baseline_layer_type + "^baseline"] = json.loads(
                        baseline_dict[baseline_title]["result"]
                    )[perf_engine]
                    compare_dict[layer_case][perf_engine + "^compare"] = perf_compare(
                        baseline=json.loads(baseline_dict[baseline_title]["result"])[perf_engine], latest=t
                    )
    return compare_dict


def perf_compare_dict(compare_list, baseline_dict, data_dict, error_list, baseline_layer_type, latest_layer_type):
    """
    生成对比dict
    :param compare_list: yaml配置中的对比项目
    :param data_dict: 待测字典
    :param baseline_dict: 基线字典
    :param error_list: list[报错子图case]
    :param baseline_layer_type: 基线子图种类，例如layercase、layerApicase
    :param latest_layer_type: 待测子图种类，例如layercase、layerApicase
    :return: 比较字典
    """
    compare_dict = {}
    for title, perf_dict in data_dict.items():  # 遍历所有子图
        if title not in error_list:
            layer_case = title.split("^", 1)[1]
            baseline_title = "^".join([baseline_layer_type, layer_case])
            layer_title = "^".join([latest_layer_type, layer_case])
            for comparing in compare_list:
                baseline_engine = comparing["baseline"]
                latest_engine = comparing["latest"]
                if layer_case not in compare_dict:
                    compare_dict[layer_case] = {}

                if baseline_engine == "ground_truth":  # 与数据库基线对比的情况
                    if baseline_title in baseline_dict and layer_title in data_dict:  # 判断数据库中的基线是否包含该子图
                        # if layer_case not in compare_dict:
                        #     compare_dict[layer_case] = {}
                        # print("layer_case is: ", layer_case)
                        compare_dict[layer_case][latest_engine + "^" + latest_layer_type] = perf_dict[latest_engine]
                        compare_dict[layer_case][latest_engine + "^" + baseline_layer_type + "^baseline"] = json.loads(
                            baseline_dict[baseline_title]["result"]
                        )[latest_engine]
                        compare_dict[layer_case][latest_engine + "^compare"] = perf_compare(
                            baseline=json.loads(baseline_dict[baseline_title]["result"])[latest_engine],
                            latest=perf_dict[latest_engine],
                        )
                    else:
                        compare_dict[layer_case][latest_engine + "^" + latest_layer_type] = perf_dict[latest_engine]
                        compare_dict[layer_case][latest_engine + "^" + baseline_layer_type + "^baseline"] = "None"
                        compare_dict[layer_case][latest_engine + "^compare"] = "None"
                else:
                    compare_dict[layer_case][latest_engine + "^" + latest_layer_type] = perf_dict[latest_engine]
                    compare_dict[layer_case][baseline_engine + "^" + baseline_layer_type] = perf_dict[baseline_engine]
                    compare_dict[layer_case][latest_engine + "^" + baseline_engine + "^compare"] = perf_compare(
                        baseline=perf_dict[baseline_engine], latest=perf_dict[latest_engine]
                    )

    return compare_dict


def perf_compare_kernel_dict(
    compare_list, baseline_dict, data_dict, error_list, baseline_layer_type, latest_layer_type
):
    """
    生成对比dict
    :param compare_list: yaml配置中的对比项目
    :param data_dict: 待测字典
    :param baseline_dict: 基线字典
    :param error_list: list[报错子图case]
    :param baseline_layer_type: 基线子图种类，例如layercase、layerApicase
    :param latest_layer_type: 待测子图种类，例如layercase、layerApicase
    :return: 比较字典
    """
    compare_dict = {}
    for title, perf_dict in data_dict.items():  # 遍历所有子图
        if title not in error_list:
            layer_case = title.split("^", 1)[1]
            # baseline_title = "^".join([baseline_layer_type, layer_case])
            # layer_title = "^".join([latest_layer_type, layer_case])
            for comparing in compare_list:
                baseline_engine = comparing["baseline"]
                latest_engine = comparing["latest"]
                if layer_case not in compare_dict:
                    compare_dict[layer_case] = {}

                if baseline_engine == "ground_truth":  # 与数据库基线对比的情况
                    # if baseline_title in baseline_dict and layer_title in data_dict:  # 判断数据库中的基线是否包含该子图
                    #     compare_dict[layer_case][latest_engine + "^" + latest_layer_type] = perf_dict[latest_engine]
                    #     compare_dict[layer_case][latest_engine + "^" + baseline_layer_type + "^baseline"]
                    #         = json.loads(
                    #         baseline_dict[baseline_title]["result"]
                    #     )[latest_engine]
                    #     compare_dict[layer_case][latest_engine + "^compare"] = perf_compare(
                    #         baseline=json.loads(baseline_dict[baseline_title]["result"])[latest_engine],
                    #         latest=perf_dict[latest_engine],
                    #     )
                    # else:
                    #     compare_dict[layer_case][latest_engine + "^" + latest_layer_type] = perf_dict[latest_engine]
                    #     compare_dict[layer_case][latest_engine + "^" + baseline_layer_type + "^baseline"] = "None"
                    #     compare_dict[layer_case][latest_engine + "^compare"] = "None"
                    pass
                else:
                    if latest_engine + "-" + "kernel_time" in perf_dict:
                        compare_dict[layer_case][
                            latest_engine + "-" + "kernel_time" + "^" + latest_layer_type
                        ] = perf_dict[latest_engine + "-" + "kernel_time"]
                    else:
                        compare_dict[layer_case][latest_engine + "-" + "kernel_time" + "^" + latest_layer_type] = "None"

                    if latest_engine + "-" + "kernel_count" in perf_dict:
                        compare_dict[layer_case][
                            latest_engine + "-" + "kernel_count" + "^" + latest_layer_type
                        ] = perf_dict[latest_engine + "-" + "kernel_count"]
                    else:
                        compare_dict[layer_case][
                            latest_engine + "-" + "kernel_count" + "^" + latest_layer_type
                        ] = "None"

                    if baseline_engine + "-" + "kernel_time" in perf_dict:
                        compare_dict[layer_case][
                            baseline_engine + "-" + "kernel_time" + "^" + baseline_layer_type
                        ] = perf_dict[baseline_engine + "-" + "kernel_time"]
                    else:
                        compare_dict[layer_case][
                            baseline_engine + "-" + "kernel_time" + "^" + baseline_layer_type
                        ] = "None"

                    if baseline_engine + "-" + "kernel_count" in perf_dict:
                        compare_dict[layer_case][
                            baseline_engine + "-" + "kernel_count" + "^" + baseline_layer_type
                        ] = perf_dict[baseline_engine + "-" + "kernel_count"]
                    else:
                        compare_dict[layer_case][
                            baseline_engine + "-" + "kernel_count" + "^" + baseline_layer_type
                        ] = "None"

                    if (
                        latest_engine + "-" + "kernel_time" in perf_dict
                        and baseline_engine + "-" + "kernel_time" in perf_dict
                    ):
                        compare_dict[layer_case][
                            latest_engine + "^" + baseline_engine + "^kernel_time_compare"
                        ] = perf_compare(
                            baseline=perf_dict[baseline_engine + "-" + "kernel_time"],
                            latest=perf_dict[latest_engine + "-" + "kernel_time"],
                        )
                    else:
                        compare_dict[layer_case][
                            latest_engine + "^" + baseline_engine + "^kernel_time_compare"
                        ] = "None"

    return compare_dict


if __name__ == "__main__":
    result = {
        "logit": [paddle.to_tensor([1.0]), paddle.to_tensor([1.0])],
        "data_grad": [paddle.to_tensor([0.0]), paddle.to_tensor([0.0])],
    }
    expect = {
        "logit": [paddle.to_tensor([0.0]), paddle.to_tensor([0.0])],
        "data_grad": [paddle.to_tensor([1.0]), paddle.to_tensor([1.0])],
    }
    res = base_compare(
        result,
        expect,
        res_name="dy_train",
        exp_name="dy_train",
        logger=Logger("PaddleLT").get_log(),
        delta=1e-10,
        rtol=1e-10,
        exc_dict={},
    )
    print("#############" * 3)
    print("res is: ", res)
