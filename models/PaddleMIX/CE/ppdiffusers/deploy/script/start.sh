#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/deploy
echo ${work_path}

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

cd ${work_path}

# controlnet
cd controlnet
(bash scripts/benchmark_paddle_deploy.sh) 2>&1 | tee ${log_dir}/controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet end***********"


# controlnet_tensorrt
(bash scripts/benchmark_paddle_deploy_tensorrt.sh) 2>&1 | tee ${log_dir}/controlnet_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet_tensorrt  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet_tensorrt  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet_tensorrt end***********"
cd ..



# ipadapter/sd15
cd ipadapter/sd15
(bash scripts/benchmark_paddle_deploy.sh) 2>&1 | tee ${log_dir}/ipadapter_sd15.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15 end***********"

# ipadapter/sd15 tensorrt
(bash scripts/benchmark_paddle_deploy_tensorrt.sh) 2>&1 | tee ${log_dir}/ipadapter_sd15_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sd15_tensorrt  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sd15_tensorrt  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sd15_tensorrt end***********"
cd ../../

# ipadapter sdxl
cd ipadapter/sdxl
(bash scripts/benchmark_paddle_deploy.sh) 2>&1 | tee ${log_dir}/ipadapter_sdxl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sdxl  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sdxl  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sdxl end***********"

# ipadapter sdxl tensorrt
(bash scripts/benchmark_paddle_deploy_tensorrt.sh) 2>&1 | tee ${log_dir}/ipadapter_sdxl_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter_sdxl_tensorrt  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter_sdxl_tensorrt  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter_sdxl_tensorrt end***********"
cd ../../

# sd15
cd sd15
(bash scripts/benchmark_paddle_deploy.sh) 2>&1 | tee ${log_dir}/sd15.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 end***********"


# sd15_tensorrt
(bash scripts/benchmark_paddle_deploy_tensorrt.sh) 2>&1 | tee ${log_dir}/sd15_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15_tensorrt  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15_tensorrt  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15_tensorrt end***********"
cd ..

# sdxl
cd sdxl
(bash scripts/benchmark_paddle_deploy.sh) 2>&1 | tee ${log_dir}/sdxl.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl end***********"


# sdxl_tensorrt
(bash scripts/benchmark_paddle_deploy_tensorrt.sh) 2>&1 | tee ${log_dir}/sdxl_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl_tensorrt  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl_tensorrt  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl_tensorrt end***********"
cd ..

# sd3 
cd sd3
(bash scripts/benchmark_paddle.sh) 2>&1 | tee ${log_dir}/sd3.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd3  success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd3  fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd3 end***********"
cd ..

echo exit_code:${exit_code}
exit ${exit_code}