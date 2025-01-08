#!/bin/bash

exit_code=0

log_dir=${root_path}/log
# 用于切换nlp的版本 
nlp_path=${root_path}/PaddleMIX

work_path=$(pwd)
echo ${work_path}

# cp change_paddlenlp_version.sh ${root_path}/PaddleMIX
bash prepare.sh

cd ${work_path}

# 遍历当前目录下的子目录
for subdir in */; do
  if [ -d "$subdir" ]; then

    # 检查子目录是否为"deploy"，如果是，则跳过(该目录需要在A100设备下跑)
    if [ "$subdir" == "infer/" ]; then
      continue
    fi

    if [ "$subdir" == "ut/" ]; then
      continue
    fi

    if [ "$subdir" == "deploy/" ]; then
      continue
    fi
    
    if [ "$subdir" == "deleted/" ]; then
      continue
    fi
    
    # 32g显存跑不了
    # if [ "$subdir" == "kandinsky2_2_text_to_image/" ]; then
    #   continue
    # fi

    # examples extra
    if [ "$subdir" == "ppdiffusers_example_test/" ]; then
      continue
    fi

    start_script_path="$subdir/start.sh"

    # 检查start.sh文件是否存在
    if [ -f "$start_script_path" ]; then
      # 执行start.sh文件，并将退出码存储在变量中
      cd $subdir
      bash start.sh
      exit_code=$((exit_code + $?))
      cd ..
    fi
  fi
done

echo "exit code: $exit_code"

# 查看结果
cat ${log_dir}/ce_res.log

exit $exit_code
