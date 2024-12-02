#!/bin/bash

exit_code=0

log_dir=${root_path}/deploy_log

work_path=$(pwd)
echo ${work_path}

bash prepare.sh

cd ${work_path}

# 遍历当前目录下的子目录
for subdir in */; do
  if [ -d "$subdir" ]; then

    if [ "$subdir" == "bk/" ]; then
      continue
    fi

    if [ "$subdir" == "controlnet/" ]; then
      continue
    fi

    if [ "$subdir" == "ipadapter_sd15/" ]; then
      continue
    fi

    if [ "$subdir" == "ipadapter_sdxl/" ]; then
      continue
    fi

    if [ "$subdir" == "sd15/" ]; then
      continue
    fi

    if [ "$subdir" == "sd3/" ]; then
      continue
    fi

    if [ "$subdir" == "sdxl/" ]; then
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

echo "*****************pip list********************"
pip list | grep paddle

echo "*****************pip list fastdeploy********************"

pip list | grep fastdeploy

python -c "
import paddle

print('Paddle Commit:', paddle.version.commit)
"

# 查看结果
cat ${log_dir}/ce_res.log


exit $exit_code
