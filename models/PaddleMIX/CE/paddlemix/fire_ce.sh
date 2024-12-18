#!/bin/bash

exit_code=0

log_dir=${root_path}/paddlemix_examples_log
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

run_list=("llava/", "qwen_vl/", "qwen2_vl/","InternVL2/", "InternLM-XComposer2/")
# 将数组转换为字符串，以便使用正则匹配
# 遍历当前目录下的子目录
for subdir in */; do
  if printf '%s\n' "${run_list[@]}" | grep -qx "$subdir"; then
    echo "子目录 ${subdir} 在列表中"
    start_script_path="$subdir/start.sh"
    # 检查start.sh文件是否存在
    if [ -f "$start_script_path" ]; then
      # 执行start.sh文件，并将退出码存储在变量中
      cd $subdir
      bash start.sh
      exit_code=$((exit_code + $?))
      cd ..
    fi
  else
    echo "子目录 ${subdir} 不在${list_str}中"
  fi
done

# 查看结果
cat ${log_dir}/ce_res.log

exit $exit_code

pip list | grep paddle




