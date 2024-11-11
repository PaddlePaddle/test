#!/usr/bin/env bash

# 最外层测试入口, 通过环境变量设定测试相关的一切信息(除了docker/whl/py以外)
export CASE_TYPE="${CASE_TYPE:-layercase}"
export CASE_DIR="${CASE_DIR:-perf_monitor}"
export TESTING="${TESTING:-yaml/dy2stcinn_eval_benchmark.yml}" # 设定测试项目配置yaml
export TESTING_MODE="${TESTING_MODE:-performance}" # 设定测试模式为性能
export PLT_MD5="${PLT_MD5:-0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a}"
export PLT_PERF_MODE="${PLT_PERF_MODE:-none}" # 设定性能测试方式, 可选参数: unit-python(独立python执行方式)
export PLT_PERF_CONTENT="${PLT_PERF_CONTENT:-layer}" # 设定性能测试内容, 可选参数: layer(端到端子图), kernel(nv工具kernel分析)
export PLT_SET_DEVICE="${PLT_SET_DEVICE:-gpu}" # 硬件
export PLT_DEVICE_ID="${PLT_DEVICE_ID:-6}"  # 设备编号
export CUDA_VISIBLE_DEVICES="${PLT_DEVICE_ID:-6}"
export FRAMEWORK="${FRAMEWORK:-paddle}"  # 框架种类
export MULTI_WORKER="${MULTI_WORKER:-0}"  # 并行数

export PLT_PYTEST_TIMEOUT="${PLT_PYTEST_TIMEOUT:-200}"  # 超时10分钟则判为失败. 设置为None则不限时

#性能测试专属环境变量
export PLT_BM_MODE="${PLT_BM_MODE:-latest}"  #基线任务为baseline, 测试任务为latest, 测试并设为新基线任务为latest_as_baseline
export PLT_BM_DB="${PLT_BM_DB:-select}"  # insert: 存入数据, 作为基线或对比; select: 不存数据, 仅对比并生成表格; non-db: 不加载数据库，仅生成表格
export PLT_BM_EMAIL="${PLT_BM_EMAIL:-False}"  # True: 发送邮件  False: 不发送邮件
export PLT_BM_REPEAT="${PLT_BM_REPEAT:-1000}"  # 性能测试重复轮次
export TIMEIT_NUM="${TIMEIT_NUM:-1}"  # timeit number数
export PLT_BM_STATIS="${PLT_BM_STATIS:-trimmean}"  # 统计策略trimmean, mean, best, best_top_k
export PLT_BM_ERROR_CHECK="${PLT_BM_ERROR_CHECK:-True}"  # True: 执行性能测试前先执行一次精度测试
export PLT_BM_PLOT="${PLT_BM_PLOT:-False}"  # True: 执行性能测试后生成性能图表

echo "wheel_url=${wheel_url}"
echo "python_ver=${python_ver}"
echo "docker_image=${docker_image}"
unset GREP_OPTIONS
env | grep CASE_
env | grep TESTING
env | grep CUDA_VISIBLE_DEVICES
env | grep FRAMEWORK
env | grep USE_PADDLE_MODEL
env | grep TIMEIT_NUM
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
# echo "MULTI_WORKER is: ${MULTI_WORKER}"
# echo "PLT_BM_MODE is: ${PLT_BM_MODE}"
# echo "PLT_BM_DB is: ${PLT_BM_DB}"
# echo "PLT_BM_REPEAT is: ${PLT_BM_REPEAT}"
# echo "PLT_BM_STATIS is: ${PLT_BM_STATIS}"
