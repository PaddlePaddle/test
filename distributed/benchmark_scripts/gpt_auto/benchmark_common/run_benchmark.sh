#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test training benchmark for a model.
# Usage：bash benchmark/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_batch_size} ${global_batch_size} ${run_mode} ${device_num} ${use_sharding}
function _set_params(){
    # 以下为用于拼接case名称的必选参数，请根据实际修改
    model_item=${model_item:-"gpt_auto"}
    global_batch_size=${global_batch_size:-64}
    fp_item=${fp_item:-"fp16"}
    run_mode=${run_mode:-"DP1-MP1-PP1-recompute"}
    device_num=${device_num:-"N1C1"}
    micro_batch_size=${micro_batch_size:-"2"}    # (必选) micro_batch_size = local_batch_size / pp_degree
    # 以下为用于数据解析的必选参数，请根据实际修改
    model_repo="PaddleNLP"
    base_batch_size=${global_batch_size}
    max_steps=${max_steps:-100}
    speed_unit="tokens/s"         # (必选)速度指标单位
    skip_steps=0                  # (必选)解析日志，跳过模型前几个性能不稳定的step
    keyword="ips:"                # (必选)解析日志，筛选出性能数据所在行的关键字
    convergence_key="loss:"       # (可选)解析日志，筛选出收敛数据所在行的关键字 如：convergence_key="loss:"
    profiling=${PROFILING:-"false"}      # (必选) Profiling  开关，默认关闭，通过全局变量传递
    # 以下为策略参数，根据具体策略需求更新数据
    dp_degree=${dp_degree:-"1"}             # (必选) dp数据并行度
    mp_degree=${mp_degree:-"1"}             # (必选) mp数据并行度
    pp_degree=${pp_degree:-"1"}             # (必选) pp数据并行度
    sharding_degree=${sharding_degree:-"1"}
    sharding_stage=${sharding_stage:-"stage1"}
    level=${level:-"o1"}               # o1|o2|o3
    local_batch_size=${local_batch_size:-"8"}    # （可选）本地batch size
    workerlog_id=${workerlog_id:-"0"}        # (可选) 指定workerlog的id，默认取0

   # 以下为通用执行命令，无特殊可不用修改
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}  # (必填) 且格式不要改动,与竞品名称对齐
    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}  # （必填） TRAIN_LOG_DIR  benchmark框架设置该参数为全局变量
    profiling_log_path=${PROFILING_LOG_DIR:-$(pwd)}  # （必填） PROFILING_LOG_DIR benchmark框架设置该参数为全局变量
    speed_log_path=${LOG_PATH_INDEX_DIR:-$(pwd)}
    train_log_file=${run_log_path}/${model_repo}_${model_name}_${device_num}_log
    mkdir -p $(dirname ${train_log_file})

    profiling_log_file=${profiling_log_path}/${model_repo}_${model_name}_${device_num}_profiling
    mkdir -p $(dirname ${profiling_log_file})

    speed_log_file=${speed_log_path}/${model_repo}_${model_name}_${device_num}_speed
    mkdir -p $(dirname ${speed_log_file})

    OUTPUT_PATH=${run_log_path}/output
}

function _train(){
    if [ -d $OUTPUT_PATH ]; then
        rm -rf $OUTPUT_PATH
    fi
    mkdir $OUTPUT_PATH

    echo "current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, model_name=${model_name}, device_num=${device_num}, is profiling=${profiling}"

    if [ ${profiling} = "true" ];then
        add_options="--profiler_options=\"batch_range=[10,20];state=GPU;tracer_option=Default;profile_path=model.profile\""
        log_file=${profiling_log_file}
    else
        add_options=""
        log_file=${train_log_file}
    fi
    
    # 单独添加的参数配置
    use_fp16=False
    if [ "fp16" = ${fp_item} ]; then
        use_fp16=True
        mix_precision_level="-o Engine.mix_precision.level=${level}"
    else
        use_fp16=False
        mix_precision_level=""
    fi

    # 根据具体执行命令修改
    train_cmd="-o Model.hidden_dropout_prob=0 \
               -o Model.attention_probs_dropout_prob=0 \
               -o Model.use_recompute=True \
               -o Global.global_batch_size=${global_batch_size} \
               -o Global.local_batch_size=${local_batch_size} \
               -o Global.micro_batch_size=${micro_batch_size} \
               -o Distributed.dp_degree=${dp_degree} \
               -o Distributed.mp_degree=${mp_degree} \
               -o Distributed.pp_degree=${pp_degree} \
               -o Distributed.sharding.sharding_degree=${sharding_degree} \
               -o Distributed.sharding.sharding_stage=${sharding_stage} \
               -o Distributed.pipeline.schedule_mode=1F1B \
               -o Engine.mix_precision.enable=${use_fp16} \
               ${mix_precision_level} \
               -o Engine.max_steps=${max_steps} \
               -o Engine.eval_freq=100000 \
               -o Profiler_auto.memory_stats=True \
               -o Engine.verbose=3 \
               -o Engine.logging_freq=1 "

    if [ ${PADDLE_TRAINER_ID} ]
    then
        PADDLE_RANK_OPTION=" --rank ${PADDLE_TRAINER_ID}"
    else
        PADDLE_RANK_OPTION=""
    fi
    # 以下为通用执行命令，无特殊可不用修改
    case ${device_num} in
    N1C1) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
            ${train_cmd}"
        workerlog_id=0
        ;;
    N1C8|N2C16|N4C32) echo "run run_mode: ${run_mode}"
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 ${PADDLE_RANK_OPTION}\
            tools/auto.py -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
            ${train_cmd}"
        workerlog_id=${workerlog_id}
        ;;
    *) echo "choose run_mode "; exit 1;
    esac
    cd ../ # 执行代码所在路径
    echo "train_cmd: ${train_cmd}  log_file: ${log_file}"
    timeout 20m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
    else
        echo -e "${model_name}, SUCCESS"
    fi
    #kill -9 `ps -ef|grep 'python'|awk '{print $2}'`
    if [ ${device_num} != "N1C1" -a -d mylog ]; then
        case_path=$PWD && cd - && mkdir -p mylog      # PaddleNLP/model_zoo/gpt-3/benchmarks
        cp -r ${case_path}/mylog/workerlog.* ./mylog/
        rm ${log_file}
        cp ${case_path}/mylog/workerlog.${workerlog_id} ${log_file}
    fi
}

export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
export FLAGS_fraction_of_gpu_memory_to_use=0.1  # 避免预分配的的显存影响实际值观测
export FLAGS_embedding_deterministic=1          # 1：关闭随机性（测试精度时为1）；0：打开随机性（测性能时为0），当前默认为1
export FLAGS_cudnn_deterministic=1              # 1：关闭随机性（测试精度时为1）；0：打开随机性（测性能时为0），当前默认为1
unset CUDA_MODULE_LOADING
env |grep FLAGS

source ${BENCHMARK_ROOT}/scripts/run_model.sh   # 在该脚本中会对符合benchmark规范的log使用analysis.py 脚本进行性能数据解析;如果不联调只想要产出训练log可以注掉本行,提交时需打开
_set_params $@
#_train       # 如果只产出训练log,不解析,可取消注释
_run     # 该函数在run_model.sh中,执行时会调用_train; 如果不联调只产出训练log可以注掉本行,提交时需打开
