#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy
echo ${work_path}

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

echo "Copying files to ${work_path}"
/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

for subdir in */; do
  if [ -d "$subdir" ]; then
    echo "Testing $subdir"
    cd "$subdir"
    echo "Copying test scripts to $subdir"
    cp -f ../test_*.sh .
    bash test_paddle.sh > ${log_dir}/${subdir}_paddle.log 2>&1
    exit_code=$((exit_code + $?))
    bash test_paddle_tensorrt.sh > ${log_dir}/${subdir}_paddle_tensorrt.log 2>&1
    exit_code=$((exit_code + $?))
    cd ..
  fi
done

cd ${work_path}/ipadapter
for subdir in */; do
  if [ -d "$subdir" ]; then
    echo "Testing $subdir"
    cd "$subdir"
    echo "Copying test scripts to $subdir"
    cp -f ../test_*.sh . 
    bash test_paddle.sh > ${log_dir}/ipadapter_${subdir}_paddle.log 2>&1
    exit_code=$((exit_code + $?))
    bash test_paddle_tensorrt.sh > ${log_dir}/ipadapter_${subdir}_paddle_tensorrt.log 2>&1
    exit_code=$((exit_code + $?))
    cd ..
  fi
done
echo exit_code:${exit_code}
exit ${exit_code}