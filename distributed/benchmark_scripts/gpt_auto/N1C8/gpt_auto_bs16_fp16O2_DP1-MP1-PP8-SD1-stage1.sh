#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# 以下为run_benchmark.sh的必选参数
param="model_item=gpt_auto "
param+="global_batch_size=16 "
param+="fp_item=fp16O2 "
param+="run_mode=DP1-MP1-PP8-SD1-stage1 "
param+="device_num=N1C8 "
param+="micro_batch_size=2 "
# 以下为run_benchmark.sh的可选参数
param+="dp_degree=1 "
param+="mp_degree=1 "
param+="pp_degree=8 "
param+="sharding_degree=1 "
param+="sharding_stage=1 "
param+="level=o2 "
param+="local_batch_size=16 "
param+="workerlog_id=7 "

cd ./benchmarks
bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/prepare.sh
# run
bash -c "${param} bash ./test_tipc/gpt/static/auto_parallel/benchmark_common/run_benchmark.sh"
