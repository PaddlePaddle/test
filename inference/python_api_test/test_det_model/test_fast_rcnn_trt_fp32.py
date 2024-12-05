# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test fast_rcnn model
"""

import os
import sys
import logging
import tarfile
import six
import wget
import pytest
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("..")
from test_case import InferenceTest


# pylint: enable=wrong-import-position


def check_model_exist():
    """
    check model exist
    """
    fast_rcnn_url = "https://paddle-qa.bj.bcebos.com/inference_model_clipped/2.2.2/detection/fast_rcnn.tgz"
    if not os.path.exists("./fast_rcnn/model.pdiparams"):
        wget.download(fast_rcnn_url, out="./")
        tar = tarfile.open("fast_rcnn.tgz")
        tar.extractall()
        tar.close()


def test_config():
    """
    test combined model config
    """
    check_model_exist()
    test_suite = InferenceTest()
    test_suite.load_config(
        model_file="./fast_rcnn/model.pdmodel",
        params_file="./fast_rcnn/model.pdiparams",
    )
    test_suite.config_test()


@pytest.mark.win
@pytest.mark.server
@pytest.mark.jetson
@pytest.mark.trt_fp32
def _test_trt_fp32_more_bz():
    """
    compared mkldnn fast_rcnn batch_size = [1] outputs with true val
    """
    check_model_exist()

    file_path = "./fast_rcnn"
    images_size = 608
    batch_size_pool = [1]
    for batch_size in batch_size_pool:

        test_suite = InferenceTest()
        test_suite.load_config(
            model_file="./fast_rcnn/model.pdmodel",
            params_file="./fast_rcnn/model.pdiparams",
        )
        images_list, images_origin_list = test_suite.get_images_npy(
            file_path, images_size, center=False, model_type="det", with_true_data=False
        )

        img = images_origin_list[0:batch_size]
        data = np.array(images_list[0:batch_size]).astype("float32")
        scale_factor_pool = []
        for batch in range(batch_size):
            scale_factor = (
                np.array([images_size * 1.0 / img[batch].shape[0], images_size * 1.0 / img[batch].shape[1]])
                .reshape((1, 2))
                .astype(np.float32)
            )
            scale_factor_pool.append(scale_factor)
        scale_factor_pool = np.array(scale_factor_pool).reshape((batch_size, 2))
        im_shape_pool = []
        for batch in range(batch_size):
            im_shape = np.array([images_size, images_size]).reshape((1, 2)).astype(np.float32)
            im_shape_pool.append(im_shape)
        im_shape_pool = np.array(im_shape_pool).reshape((batch_size, 2))
        input_data_dict = {"im_shape": im_shape_pool, "image": data, "scale_factor": scale_factor_pool}
        output_data_dict = test_suite.get_truth_val(input_data_dict, device="gpu")

        del test_suite  # destroy class to save memory

        test_suite2 = InferenceTest()
        test_suite2.load_config(
            model_file="./fast_rcnn/model.pdmodel",
            params_file="./fast_rcnn/model.pdiparams",
        )
        test_suite2.trt_more_bz_test(
            input_data_dict,
            output_data_dict,
            repeat=1,
            delta=3e-5,
            precision="trt_fp32",
            min_subgraph_size=1,
            dynamic=True,
            auto_tuned=True,
        )

        del test_suite2  # destroy class to save memory
