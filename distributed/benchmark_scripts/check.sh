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
####################################
export dir_name=$1  # /path/to/demo-model_name

# 判断shellcheck是否安装
echo "==============shellcheck=============="
if command -v shellcheck &> /dev/null; then  
    echo "shellcheck is installed"  
else  
    echo "shellcheck is not installed, install it now"
    echo "apt install shellcheck or yum install shellcheck, exit 1"
    exit 1
fi

check_case_name(){
    file=$1
    echo "=========校验文件名称格式========="
    file_temp=${file##*/}
    file_name=${file_temp%.sh}
    file_model_item=${file_name%%_bs*}
    file_global_batch_size=$(echo "$file_name" | grep -oP '_bs\K\d+')
    file_fp_item=$(echo "$file_name" | grep -oP 'bf[^_]*|fp[^_]*') 
    file_run_mode=$(echo "$file_name" | sed 's/.*_//')
    
    model_item=$(cat $file|grep -oP 'model_item=\K[^"]*' | sed 's/ *$//')
    global_batch_size=$(cat $file|grep -oP 'global_batch_size=\K\d+' | sed 's/ *$//')
    fp_item=$(cat $file|grep -oP 'fp_item=\K[^"]*' | sed 's/ *$//')
    run_mode=$(cat $file|grep -oP 'run_mode=\K[^"]*' | sed 's/ *$//')
    model_name=${model_item}_bs${global_batch_size}_${fp_item}_${run_mode}

    if [[ $file_name != "$model_name" ]]; then
        echo "异常退出,文件名与model_name拼接结果不一致!!!文件名:$file_name, model_name拼接结果:$model_name"
        exit 1
    fi
    if [[ $file_model_item != $model_item ]]; then
        echo "异常退出,model_item不一致!!!文件名的model_item:$file_model_item, 文件名内容的model_item:$model_item"
        exit 1
    fi
    if [[ $file_global_batch_size != $global_batch_size ]]; then
        echo "异常退出,global_batch_size不一致!!!文件名的global_batch_size:$file_global_batch_size, 文件名内容的global_batch_size:$global_batch_size"
        exit 1
    fi
    # fp_item为空时，不校验
    if [[ $file_fp_item != "$fp_item" ]] && [[ $fp_item != null ]]; then
        echo "异常退出,fp_item不一致!!!文件名的fp_item:$file_fp_item, 文件名内容的fp_item:$fp_item"
        exit 1
    fi
    if [[ $file_run_mode != "$run_mode" ]]; then
        echo "异常退出,run_mode不一致!!!文件名的run_mode:$file_run_mode, 文件名内容的run_mode:$run_mode"
        exit 1
    fi
    echo "=========校验文件名称格式成功========="
}
check_param_mode(){
    file=$1
    file_content=$(cat "$file")  
    if [[ $file_content == *'param+='* ]]; then
        echo "匹配param模式校验"
        if grep -qE '^param="[^"]+' "$file"; then
            echo "param=存在"
        else  
            echo "异常退出,文件内容不包含'param='"  
            exit 1  
        fi
        # 遍历文件的每一行，确认包括param的行为空格结束
        while IFS= read -r line; do  
            # 检查行是否包含 param+=  
            if [[ "$line" == *"param+="* ]] || [[ "$line" == *"param="* ]]; then  
                # 检查行是否包含空格和双引号  
                if [[ "$line" == *" "* ]] && [[ "$line" == *'"'* ]]; then  
                    continue
                else
                    echo "异常退出, param=|+=结尾不以空格和双引号结尾：$line"  
                    exit 1  
                fi  
            fi  
        done < "$file"
        echo "param=|+=均以空格和双引号结尾，符合预期"
    fi
}
check_run_mode(){
    file=$1
    run_mode=$(cat $file|grep -oP 'run_mode=\K[^"]*' | sed 's/ *$//')
    run_benchmark=$(dirname "$file")/../benchmark_common/run_benchmark.sh
    if grep -qE "{run_mode} in" "$run_benchmark"; then 
        echo "匹配run_mode模式校验"
        if grep -qE "$run_mode" "$run_benchmark"; then
            echo "run_mode匹配成功"
        else
            echo "异常退出，文件内容不包含$run_mode"
            exit 1
        fi
    fi
}

check_args(){
    file=$1
    if ! grep -qP '^model_item=[^/]*$' "$file"; then  
        echo "异常退出,文件内容不包含model_item或model_item包含/字符"
        exit 1
    fi
    if ! grep -qP '^global_batch_size=' "$file"; then  
        echo "异常退出,文件内容不包含global_batch_size"
        exit 1
    fi
    if ! grep -qP '^fp_item=' "$file"; then
        echo "异常退出,文件内容不包含fp_item"
        exit 1
    fi
    if ! grep -qP '^run_mode=' "$file"; then
        echo "异常退出,文件内容不包含run_mode"
        exit 1
    fi
    model_name_str=$(cat $file|grep -oP 'model_item=\K[^"]*' | sed 's/ *$//')
}

# 定义递归函数，用于遍历文件夹  
traverse_folder() {  
    local folder=$1
  
    # 遍历当前文件夹下的所有文件和子文件夹  
    for item in "$folder"/*; do  
        # 检查是否为文件夹  
        if [[ -d "$item" ]]; then  
            traverse_folder "$item"  # 递归调用自身，遍历子文件夹  
        else  
            echo "文件: ${item}" 
            # shellcheck
            shellcheck --format=gcc ${item} | grep -v '^$' | grep -v '^#'| grep error 
            if [[ ${item} == *"N"*"C"*".sh" ]]; then
                # 校验文件名称格式
                check_case_name ${item}
                # 校验param模式
                check_param_mode ${item}
                # 校验run_mode匹配
                check_run_mode ${item}
            elif [[ ${item} == *"run_benchmark.sh" ]]; then
                check_args ${item}
            else
                echo "其他脚本人工check"
            fi

        fi  
    done  
}  
  
# 调用递归函数，开始遍历  
traverse_folder "$dir_name"