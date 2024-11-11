#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export CASE_TYPE="${CASE_TYPE:-layercase}"
export CASE_DIR="${CASE_DIR:-sublayer1000}"
export TESTING="${TESTING:-yaml/dy^dy2stcinn_eval.yml}" # 设定测试项目配置yaml
export TESTING_MODE="${TESTING_MODE:-precision}" # 设定测试模型为精度
export PLT_MD5="${PLT_MD5:-0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a}"
export PLT_PERF_CONTENT="${PLT_PERF_CONTENT:-layer}"
export PLT_SET_DEVICE="${PLT_SET_DEVICE:-gpu}"
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-6}"
export CUDA_VISIBLE_DEVICES="${PLT_DEVICE_ID:-6}"
export FRAMEWORK="${FRAMEWORK:-paddle}"
export USE_PADDLE_MODEL="${USE_PADDLE_MODEL:-None}"  # 设定是否使用paddle模型库, 可选PaddleOCR
export MULTI_WORKER="${MULTI_WORKER:-0}"
export MULTI_DOUBLE_CHECK="${MULTI_DOUBLE_CHECK:-True}"

export PLT_PYTEST_TIMEOUT="${PLT_PYTEST_TIMEOUT:-600}"  # 超时10分钟则判为失败. 设置为None则不限时
export PLT_SPEC_USE_MULTI="${PLT_SPEC_USE_MULTI:-False}"  # 开启动态InputSpec搜索遍历
export PLT_SAVE_SPEC="${PLT_SAVE_SPEC:-False}"  # 是否保存InputSpec搜索遍历结果
export PLT_SAVE_GT="${PLT_SAVE_GT:-False}"  # 是否保存精度ground truth, 也就是plt_gt
export PLT_GT_UPLOAD_URL="${PLT_GT_UPLOAD_URL:-None}"  # plt_gt的上传路径, paddle-qa/PaddleLT/PaddleLTGroundTruth/latest
export PLT_GT_DOWNLOAD_URL="${PLT_GT_DOWNLOAD_URL:-None}"  # plt_gt的下载url, https://paddle-qa.bj.bcebos.com/PaddleLT/PaddleLTGroundTruth/latest/gpu

# 精度结果入库
export PLT_BM_MODE="${PLT_BM_MODE:-baseline}"  #基线任务为baseline, 测试任务为latest, 测试并设为新基线任务为latest_as_baseline
export PLT_BM_DB="${PLT_BM_DB:-non-db}"  # insert: 存入数据, 作为基线或对比; select: 不存数据, 仅拉取之前结果; non-db: 不加载数据库

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

# echo "wheel_url is: ${wheel_url}"
# echo "python_ver is: ${python_ver}"
# echo "CASE_TYPE is: ${CASE_TYPE}"
# echo "CASE_DIR is: ${CASE_DIR}"
# echo "TESTING is: ${TESTING}"
# echo "CUDA_VISIBLE_DEVICES is: ${CUDA_VISIBLE_DEVICES}"

# echo "TESTING_MODE is: ${TESTING_MODE}"
# echo "PLT_SET_DEVICE is: ${PLT_SET_DEVICE}"
# echo "PLT_DEVICE_ID is: ${PLT_DEVICE_ID}"
# echo "FRAMEWORK is: ${FRAMEWORK}"
# echo "USE_PADDLE_MODEL is: ${USE_PADDLE_MODEL}"
# echo "MULTI_WORKER is: ${MULTI_WORKER}"
# echo "MULTI_DOUBLE_CHECK is: ${MULTI_DOUBLE_CHECK}"

# echo "PLT_PYTEST_TIMEOUT is: ${PLT_PYTEST_TIMEOUT}"
# echo "PLT_SPEC_USE_MULTI is: ${PLT_SPEC_USE_MULTI}"
# echo "PLT_SAVE_SPEC is: ${PLT_SAVE_SPEC}"
# echo "PLT_SAVE_GT is: ${PLT_SAVE_GT}"
# echo "PLT_GT_UPLOAD_URL is: ${PLT_GT_UPLOAD_URL}"
# echo "PLT_GT_DOWNLOAD_URL is: ${PLT_GT_DOWNLOAD_URL}"
