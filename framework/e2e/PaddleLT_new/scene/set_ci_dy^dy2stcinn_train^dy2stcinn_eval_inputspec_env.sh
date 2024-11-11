#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export python_ver=python3.10
export CASE_TYPE=layercase
export CASE_DIR=sublayer1000
export TESTING=yaml/CI_dy^dy2stcinn_train^dy2stcinn_eval_inputspec.yml # 设定测试项目配置yaml
export TESTING_MODE=precision_multi_gpu # 设定测试模型为精度
export PLT_SET_DEVICE=gpu
export PLT_GET_NV_MEMORY=False
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-0,1}"
export CUDA_VISIBLE_DEVICES="${PLT_DEVICE_ID:-0,1}"
export FRAMEWORK=paddle
export MULTI_WORKER=12
export MULTI_DOUBLE_CHECK=True

export PLT_PYTEST_TIMEOUT=200  # 超时10分钟则判为失败. 设置为None则不限时
export PLT_SPEC_USE_MULTI=False  # 开启动态InputSpec搜索遍历
export PLT_SAVE_SPEC=False  # 是否保存InputSpec搜索遍历结果
export PLT_SAVE_GT=False  # 是否保存精度ground truth, 也就是plt_gt
export PLT_GT_UPLOAD_URL=None  # plt_gt的上传路径, paddle-qa/PaddleLT/PaddleLTGroundTruth/latest
export PLT_GT_DOWNLOAD_URL=None  # plt_gt的下载url, https://paddle-qa.bj.bcebos.com/PaddleLT/PaddleLTGroundTruth/latest/gpu

export FLAGS_prim_forward_blacklist=pd_op.dropout
export FLAGS_enable_pir_api=1
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=true
export FLAGS_cinn_new_group_scheduler=1
export FLAGS_group_schedule_tiling_first=1
export FLAGS_cinn_bucket_compile=True
export FLAGS_cinn_compile_with_nvrtc=True
export FLAGS_nvrtc_compile_to_cubin=True
export FLAGS_support_reduce_stride_read=1

echo "wheel_url is: ${wheel_url}"
echo "python_ver is: ${python_ver}"
echo "CASE_TYPE is: ${CASE_TYPE}"
echo "CASE_DIR is: ${CASE_DIR}"
echo "TESTING is: ${TESTING}"
echo "CUDA_VISIBLE_DEVICES is: ${CUDA_VISIBLE_DEVICES}"
echo "MULTI_DOUBLE_CHECK is: ${MULTI_DOUBLE_CHECK}"

echo "TESTING_MODE is: ${TESTING_MODE}"
echo "PLT_SET_DEVICE is: ${PLT_SET_DEVICE}"
echo "PLT_DEVICE_ID is: ${PLT_DEVICE_ID}"
echo "FRAMEWORK is: ${FRAMEWORK}"
echo "MULTI_WORKER is: ${MULTI_WORKER}"

echo "PLT_PYTEST_TIMEOUT is: ${PLT_PYTEST_TIMEOUT}"
echo "PLT_SPEC_USE_MULTI is: ${PLT_SPEC_USE_MULTI}"
echo "PLT_SAVE_SPEC is: ${PLT_SAVE_SPEC}"
echo "PLT_SAVE_GT is: ${PLT_SAVE_GT}"
echo "PLT_GT_UPLOAD_URL is: ${PLT_GT_UPLOAD_URL}"
echo "PLT_GT_DOWNLOAD_URL is: ${PLT_GT_DOWNLOAD_URL}"
