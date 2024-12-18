#!/bin/bash

exit_code=0

work_path=$(pwd)
echo ${work_path}

cd ${root_path}/PaddleMIX/

python -m pip install --upgrade pip
pip install -r requirements.txt

python -m pip install --upgrade pip
pip install -r requirements.txt
# 测试环境需要
pip install pexpect
pip install einops
cd ppdiffusers
pip install -e .
cd ..
pip install -e .


cd ${work_path}

run_list=("llava", "qwen_vl", "qwen2_vl","InternVL2", "InternLM-XComposer2")
# 遍历当前目录下的子目录
for subdir in */; do
  if [[ " ${run_list[*]} " =~ " $subdir " ]]; then
    start_script_path="$subdir/start.sh"
    if [ -f "$start_script_path" ]; then
      # 执行start.sh文件，并将退出码存储在变量中
      cd $subdir
      bash start.sh
      exit_code=$((exit_code + $?))
      cd ..
    fi
  else
    echo "no start.sh in $subdir"
  fi
done

# 查看结果
cat ${log_dir}/ce_res.log

exit $exit_code

pip list | grep paddle




