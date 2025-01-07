#!/usr/bin/env bash

# 最外层执行脚本, 不可用于CI
test_scene=$1
source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./set_docker_env.sh # 设定docker环境相关参数

export docker_name=${DOCKER_NAME:-PaddleLayerTest_${AGILE_PIPELINE_BUILD_NUMBER}}
docker container ls -a --filter "name=${docker_name}" --format "{{.ID}}" | xargs -r docker rm -f

mkdir root_tmp

nvidia-docker run --rm -i --name ${docker_name} --privileged --shm-size=128g --net=host \
  -w /workspace \
  -v $PWD:/workspace \
  -v root_tmp:/root \
  -e "AK=${AK}" -e "SK=${SK}" \
  -e "http_proxy=${http_proxy}" \
  -e "https_proxy=${https_proxy}" \
  -e "no_proxy=bcebos.com" \
  -e "PLT_DEVICE_ID=${PLT_DEVICE_ID}" \
  -e "python_ver=${python_ver}" \
  -e "wheel_url=${wheel_url}" \
  -e "AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID}" \
  ${docker_image} /bin/bash -c "
ldconfig;

source ./PTSTools/tools/set_env/set_env.sh ${PTS_ENV_VARS}  # 设定PTS环境变量
source ./${test_scene}
${python_ver} -m pip install -r requirement.txt
${python_ver} -m pip install -r ./PTSTools/LogParseUpload/requirement.txt;
cp -r ./PTSTools/Uploader/apibm_config.yml .
${python_ver} -m pip install ${wheel_url}
${python_ver} run.py

cp -r report ./PTSTools/LogParseUpload;
cd ./PTSTools/LogParseUpload;
unset http_proxy && unset https_proxy;

echo 以下开始pts的pytest-allure回调
${python_ver} upload_plt.py --file_path report --id ${pts_id} --status '成功'
"
