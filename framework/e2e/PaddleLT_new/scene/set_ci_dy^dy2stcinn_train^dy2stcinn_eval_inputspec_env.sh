#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export python_ver=python3.10
export CASE_TYPE=layercase
export CASE_DIR=sublayer1000
export TESTING=yaml/CI_dy^dy2stcinn_train^dy2stcinn_eval_inputspec.yml # 设定测试项目配置yaml
export TESTING_MODE=precision_multi_gpu # 设定测试模型为精度
export PLT_MD5=4ed830834040691d336abc8d4bb9f7f7
export PLT_PERF_CONTENT="${PLT_PERF_CONTENT:-layer}"
export PLT_SET_DEVICE=gpu
export PLT_GET_NV_MEMORY=False
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-0,1}"
export CUDA_VISIBLE_DEVICES="${PLT_DEVICE_ID:-0,1}"
export FRAMEWORK=paddle
export USE_PADDLE_MODEL="${USE_PADDLE_MODEL:-None}"
export MULTI_WORKER=12
export MULTI_DOUBLE_CHECK=True

export PLT_PYTEST_TIMEOUT=200  # 超时200秒则判为失败. 设置为None则不限时
export PLT_SPEC_USE_MULTI=False  # 开启动态InputSpec搜索遍历
export PLT_SAVE_SPEC=False  # 是否保存InputSpec搜索遍历结果
export PLT_SAVE_GT=False  # 是否保存精度ground truth, 也就是plt_gt
export PLT_GT_UPLOAD_URL=None  # plt_gt的上传路径, paddle-qa/PaddleLT/PaddleLTGroundTruth/latest
export PLT_GT_DOWNLOAD_URL=None  # plt_gt的下载url, https://paddle-qa.bj.bcebos.com/PaddleLT/PaddleLTGroundTruth/latest/gpu

# 精度结果入库
export PLT_BM_MODE="${PLT_BM_MODE:-baseline}"
export PLT_BM_DB=insert

export MIN_GRAPH_SIZE=0
export FLAGS_prim_all=true
export FLAGS_use_cinn=1
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_forward_blacklist=pd_op.dropout

echo "wheel_url=${wheel_url}"
echo "python_ver=${python_ver}"
echo "docker_image=${docker_image}"
unset GREP_OPTIONS
env | grep CASE_
env | grep TESTING
env | grep CUDA_VISIBLE_DEVICES
env | grep FRAMEWORK
env | grep USE_PADDLE_MODEL
env | grep MULTI_
env | grep PLT_
env | grep FLAGS_
