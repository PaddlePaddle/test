#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy
echo ${work_path}

gather_file_path=${root_path}/deploy_gather_file

if [ ! -d ${gather_file_path} ]; then
    mkdir ${gather_file_path}
    echo "create dir ${gather_file_path}"
fi

echo "Copying filles to ${work_path}"
/bin/cp -rf ./* ${work_path}/

exit_code=0

cd ${work_path}
# 遍历所有子目录
find . -type d \( -name "results-paddle" -o -name "results-paddle-fp16" -o -name "results-paddle_tensorrt" -o -name "results-paddle_tensorrt-fp16" \) | while read dir; do
    # 提取父目录路径作为子目录名
    PARENT_DIR=$(basename "$(dirname "$dir")")

    # 创建目标嵌套目录
    mkdir -p "$gather_file_path/$PARENT_DIR"


    # 复制：保留原文件夹，复制到目标目录
    cp -r "$dir" "$TARGET_DIR/$PARENT_DIR/"
done

