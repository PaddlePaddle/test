#!/bin/bash

log_dir=${root_path}/log

exit_code=0

work_path=$(pwd)
echo ${work_path}

rm -rf /root/.paddlemix/models/paddlemix/EVA/
rm -rf /root/.paddlenlp/models/paddlemix/EVA

cd ${root_path}
mkdir data
cd data
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/ILSVRC2012/imagenet-val.tar
tar -xvf imagenet-val.tar

cd ${root_path}/
mkdir dataset
cd dataset
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/ILSVRC2012/ILSVRC2012_tiny.tar
tar -xvf ILSVRC2012_tiny.tar

cd ${root_path}/PaddleMIX/

# export http_proxy=${proxy}
# export https_proxy=${proxy}
python -m pip install --upgrade pip
pip install -r requirements.txt
# 测试环境需要
pip install pexpect
pip install einops
cd ppdiffusers
pip install -e .
cd ..
pip install -e .
# unset http_proxy
# unset https_proxy

cd ${work_path}

# 遍历当前目录下的子目录
for subdir in */; do
  if [ -d "$subdir" ]; then
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
