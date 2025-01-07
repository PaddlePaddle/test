set +x;
pwd;

### 加载tools/linux_env_info.sh文件
if [ -e linux_env_info.sh ];then
    rm -rf linux_env_info.sh
fi
# wget -q https://raw.githubusercontent.com/PaddlePaddle/PaddleTest/develop/tools/linux_env_info.sh --no-proxy
# 临时使用
wget -q https://paddle-qa.bj.bcebos.com/PaddleMT/linux_env_info.sh --no-proxy
source ./linux_env_info.sh
set +e

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

######################## 定义变量 ########################
# AGILE_PIPELINE_NAME 格式类似: PaddleClas-Linux-Cuda102-Python37-P0-Develop      (默认使用 Ubuntu 和用户保持一致)
# AGILE_PIPELINE_NAME 格式类似: PaddleClas-Linux-Cuda102-Python37-P0-Develop-Centos  (额外 Centos 和编包保持一致)
#其它内容或者可能不一致的不要随意加 "-", 下面是按照 "-" split 按序号填入的

#repo的名称
export reponame=${reponame:-"`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $1}')`"}

#模型列表文件 , 固定路径及格式为 tools/reponame_优先级_list   优先级P2有多个用P21、P22  中间不用"-"划分, 防止按 "-" split 混淆
export models_file=${models_file:-"tools/${reponame}_`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $5}')`_list"}
export models_list=${models_list:-None} #模型列表

#指定case操作系统
if [[ ${AGILE_PIPELINE_NAME} =~ "-Linux-" ]];then
    export system=${system:-"linux"}   # linux windows windows_cpu mac 与yaml case下字段保持一致
elif [[ ${AGILE_PIPELINE_NAME} =~ "-LinuxConvergence-" ]];then
    export system=${system:-"linux_convergence"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-Windows-" ]];then
    export system=${system:-"windows"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-WindowsCPU-" ]];then
    export system=${system:-"windows_cpu"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-Mac-" ]];then
    export system=${system:-"mac"}
else
    if [[ ${system} ]];then
        system=${system}
        ## 防止环境不匹配，不设置默认的 system
        # export system=${system:-"linux"}
    else
        echo "do not set system   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi

#指定python版本
export Python_version=${Python_version:-"`(echo ${AGILE_PIPELINE_NAME}|awk -F 'Python' '{print $2}'|awk -F '-' '{print $1}')`"}


#指定docker镜像
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda102
    else
        linux_env_info_main get_docker_images Ubuntu Cuda102
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda112
    else
        linux_env_info_main get_docker_images Ubuntu Cuda112
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda116
    else
        linux_env_info_main get_docker_images Ubuntu Cuda116
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda117
    else
        linux_env_info_main get_docker_images Ubuntu Cuda117
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda118" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda118
    else
        linux_env_info_main get_docker_images Ubuntu Cuda118
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda120" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        linux_env_info_main get_docker_images Centos Cuda120
    else
        linux_env_info_main get_docker_images Ubuntu Cuda120
    fi
else
    if [[ ${Image_version} ]];then
        Image_version=${Image_version}
        ## 防止环境不匹配，不设置默认的 Image_version
        # export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev"}
    else
        echo "do not set Image_version   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi

#约定覆盖的几条流水线
#指定whl包, 暂时用night develop的包
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python36" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-linux_x86_64.whl"}
        export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-7.1.3.4"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-linux_x86_64.whl"}
        export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-7.1.3.4"}

    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python37" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda102 Python37 Develop
        # cudnn7用的是cudnn8的预测库的包
        export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-7.0.0.11"}

        #230223 stride test
        # export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/xieyunshen/TempPRBuild/50444/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"}
    else
        linux_env_info_main get_wheel_url Cuda102 Python37 Release
        # cudnn7用的是cudnn8的预测库的包
        export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-7.0.0.11"}

    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python38" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda112 Python38 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    else
        linux_env_info_main get_wheel_url Cuda112 Python38 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python311" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda112 Python311 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    else
        linux_env_info_main get_wheel_url Cuda112 Python311 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-Centos-Gcc82-Cuda112-Cudnn82-Trt8034-Py38-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python312" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda112 Python312 Develop ON
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    else
        linux_env_info_main get_wheel_url Cuda112 Python312 Release ON
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.0.3.4"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python39" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda116 Python39 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-inference-lib.bj.bcebos.com/develop/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.0.6"}
    else
        linux_env_info_main get_wheel_url Cuda116 Python39 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-inference-lib.bj.bcebos.com/release/2.5/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.0.6"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python310" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url_covall Cuda116 Python310 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-inference-lib.bj.bcebos.com/develop/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.0.6"}
    else
        linux_env_info_main get_wheel_url_covall Cuda116 Python310 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-inference-lib.bj.bcebos.com/release/2.5/cxx_c/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.6_cudnn8.4.0-trt8.4.0.6/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.0.6"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python310" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda117 Python310 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.2.4"}
    else
        linux_env_info_main get_wheel_url Cuda117 Python310 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.4.2.4"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda118" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python310" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda118 Python310 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.5.1.7"}
    else
        linux_env_info_main get_wheel_url Cuda118 Python310 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.5.1.7"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda120" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python311" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        linux_env_info_main get_wheel_url Cuda120 Python311 Develop ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.6.1.6"}
    else
        linux_env_info_main get_wheel_url Cuda120 Python311 Release ON
        # export paddle_inference=${paddle_inference:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py39-Compile/latest/paddle_inference.tgz"}
        export TENSORRT_DIR=${TENSORRT_DIR:-"/usr/local/TensorRT-8.6.1.6"}
    fi
else
    if [[ ${paddle_whl} ]];then
        paddle_whl=${paddle_whl}
        ## 防止环境不匹配，不设置默认的paddle_whl
        # if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        #     export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl"}
        # else
        #     export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl"}
        # fi
    else
        echo "do not set paddle_whl   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi


#### 可能要改的参数
export step=${step:-train:all+eval:all+infer:all+export:all+predict:all}  #阶段 demo:train:multi,single+eval:trained,pretrained, 所有流水线都要自己改
export branch=${branch:-develop}    # repo的分支，大部分为develop，如果有master dygraph等注意设置!!
export mode=${mode:-function}   #function只验证功能是否正常  precision验证功能&小数据集精度
export timeout=${timeout:-3600}   #timeout 为超时取消时间, 单位为秒
export docker_flag=${docker_flag:-} # 如果北京集群cce环境为False，自己的开发机&release机器不用设置

#### 建议不改的参数
export use_build=${use_build:-yes}  #流水线默认为yes，是否在main中执行环境部署
export get_repo=${get_repo:-wget} #现支持10个库，需要的话可以加，wget快很多
export set_cuda=${set_cuda:-} #预先不设置   #设置显卡号，流水线不用设置，后面有通过release_01后缀判断

#### 数据软链使用
export dataset_org=${dataset_org:-None}     #如需软链数据基于根目录的原始地址 demo: /ssd2/ce_data
export dataset_target=${dataset_target:-None}   #如需软链数据基于reponame的目标地址 demo: data/flower102

#### 全局设置的参数
export no_proxy=${no_proxy:-}
export http_proxy=${http_proxy:-}   # 代理在效率云全局变量设置
export AGILE_PIPELINE_CONF_ID=${AGILE_PIPELINE_CONF_ID}   #效率云依赖参数
export AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID} #效率云依赖参数
export AGILE_JOB_BUILD_ID=${AGILE_JOB_BUILD_ID}   #效率云依赖参数
export AGILE_WORKSPACE=${AGILE_WORKSPACE}   #效率云依赖参数
export REPORT_SERVER_PASSWORD=${REPORT_SERVER_PASSWORD}   #上传全局变量

#### 根据PaddleTest & 框架名称决定的参数
export CE_version_name=${CE_version_name:-TestFrameWork}    #与测试框架的名称一致
export models_name=${models_name:-models_restruct}  #后面复制使用，和模型库的父路径目录一致（后续改为models）

#### 二分定位使用
export binary_search_flag=${binary_search_flag:-False}  #True表示在使用二分定位, main中一些跳出方法不生效
# 使用大模型分析模型日志
# export is_analysis_logs=${is_analysis_logs:-True}
export analysis_case=${analysis_case:-None}
export use_data_cfs=${use_data_cfs:-False}  #False表示不用cfs挂载
export plot=${plot:-False}  #False表示不自动绘图
export c_plus_plus_predict=${c_plus_plus_predict:-False}  #False表示不配置 C++预测库
export PaddleX=${PaddleX:-None}

#cinn编译器
export FLAGS_use_cinn=${FLAGS_use_cinn:-0}
export FLAGS_prim_all=${FLAGS_prim_all:-false}
# new ir
export FLAGS_enable_pir_in_executor=${FLAGS_enable_pir_in_executor:-0}
export FLAGS_enable_pir_api=${FLAGS_enable_pir_api:-0}
# paddleSOT
export ENABLE_FALL_BACK=${ENABLE_FALL_BACK:-true}
export MIN_GRAPH_SIZE=${MIN_GRAPH_SIZE:-10}
export FLAGS_pir_subgraph_saving_dir=${FLAGS_pir_subgraph_saving_dir:-}
export FLAGS_enable_pir_api=${FLAGS_enable_pir_api:-False}
export SOT_EXPORT_FLAG=${SOT_EXPORT_FLAG:-False} #240123 针对泽宇项目设为True
export GLOG_vmodule=${GLOG_vmodule:-}

# 显存复用
export FLAGS_use_auto_growth_v2=${FLAGS_use_auto_growth_v2:-0}

######################## 开始执行 ########################
####    测试框架下载    #####
wget -q ${CE_Link} --no-proxy #需要全局定义
unzip -P ${CE_pass} ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
export http_proxy=${http_proxy}
export https_proxy=${http_proxy}
export no_proxy=${no_proxy}
export AK=${AK} #使用bos_new上传需要
export SK=${SK}
export api_key=${api_key}
export bce_whl_url=${bce_whl_url}
set -x;

#输出参数验证
echo "@@@reponame: ${reponame}"
echo "@@@models_list: ${models_list}"
echo "@@@models_file: ${models_file}"
echo "@@@system: ${system}"
echo "@@@Python_version: ${Python_version}"
echo "@@@Image_version: ${Image_version}"
echo "@@@paddle_whl: ${paddle_whl}"
echo "@@@step: ${step}"
echo "@@@branch: ${branch}"
echo "@@@mode: ${mode}"
echo "@@@docker_flag: ${docker_flag}"
echo "@@@timeout: ${timeout}"
echo "@@@binary_search_flag: ${binary_search_flag}"
# echo "@@@is_analysis_logs: ${is_analysis_logs}"
echo "@@@analysis_case: ${analysis_case}"
echo "@@@use_data_cfs: ${use_data_cfs}"
echo "@@@plot: ${plot}"
echo "@@@c_plus_plus_predict: ${c_plus_plus_predict}"
echo "@@@paddle_inference: ${paddle_inference}"
echo "@@@TENSORRT_DIR: ${TENSORRT_DIR}"

echo "@@@FLAGS_use_cinn: ${FLAGS_use_cinn}"
echo "@@@FLAGS_prim_all: ${FLAGS_prim_all}"
echo "@@@FLAGS_pir_subgraph_saving_dir: ${FLAGS_pir_subgraph_saving_dir}"

####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #如果预先下载直接mv
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

#复制模型相关文件到指定位置
if [ $PaddleX = True ];then
  echo "PaddleX is True"
  cp -r ./task/${models_name}/PaddleX/.  ./${CE_version_name}/
else
  echo "PaddleX is False"
  cp -r ./task/${models_name}/${reponame}/.  ./${CE_version_name}/
fi

ls ./${CE_version_name}/
cd ./${CE_version_name}/

##如果预先模型库下载直接mv, 方便二分是checkout 到某个commit进行二分
if [[ -d "../../${reponame}" ]];then  #前面cd 了 2次所以使用 ../../
    cp -r ../../${reponame} .
    echo "因为 ${reponame} 在根目录存在 使用预先clone或wget的 ${reponame}"
fi

# PTS
if [[ -d "../../PTSTools" ]];then
    cp -r ../../PTSTools .
    echo "Get PTSTools"
fi

####根据agent制定对应卡，记得起agent时文件夹按照release_01 02 03 04名称
if  [[ "${set_cuda}" == "" ]] ;then  #换了docker启动的方式，使用默认制定方式即可，SET_MULTI_CUDA参数只是在启动时使用
    tc_name=`(echo $PWD|awk -F 'xly/' '{print $2}'|awk -F '/' '{print $1}')`
    echo "teamcity path:" $tc_name
    if [ $tc_name == "release_02" ];then
        echo release_02
        export set_cuda=2,3;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia2 | awk '{print $0}' | xargs kill -9
            fuser -v /dev/nvidia3 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_03" ];then
        echo release_03
        export set_cuda=4,5;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia4 | awk '{print $0}' | xargs kill -9
            fuser -v /dev/nvidia5 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_04" ];then
        echo release_04
        export set_cuda=6,7;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia6 | awk '{print $0}' | xargs kill -9
            fuser -v /dev/nvidia7 | awk '{print $0}' | xargs kill -9
        fi
    else
        echo release_01
        export set_cuda=0,1;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia0 | awk '{print $0}' | xargs kill -9
            fuser -v /dev/nvidia1 | awk '{print $0}' | xargs kill -9
        fi
    fi
else
    echo already seted CUDA_id  #这里需要再细化下，按下面的方法指定无用，直接默认按common中指定0,1卡了
    export set_cuda=${set_cuda}
fi

if [[ "${docker_flag}" == "" ]]; then

    echo "before set_cuda: $set_cuda" #在docker外更改set_cuda从0开始计数, 在cce或线下已在docker中不执行
    export set_cuda_back=${set_cuda};
    array=(${set_cuda_back//,/ });
    set_cuda=0;
    for((i=1;i<${#array[@]};i++));
    do
    export set_cuda=${set_cuda},${i};
    done
    echo "after set_cuda: $set_cuda"

    ####创建docker
    set +x;

    # # 拉取更新镜像
    docker pull ${Image_version}

    docker_name="ce_${AGILE_PIPELINE_NAME}_${AGILE_JOB_BUILD_ID}" #AGILE_JOB_BUILD_ID以每个流水线粒度区分docker名称
    function docker_del()
    {
    echo "begin kill docker"
    docker rm -f ${docker_name}
    echo "end kill docker"
    }
    trap 'docker_del' SIGTERM
    ## 使用修改之前的set_cuda_back
    NV_GPU=${set_cuda_back} nvidia-docker run -i \
        --name=${docker_name} --net=host --cap-add=SYS_ADMIN \
        --shm-size=128G \
        -v $(pwd):/workspace \
        -v /ssd2:/ssd2 \
        -v /mnt:/mnt \
        -e AK=${AK} \
        -e SK=${SK} \
        -e CFS_IP=${CFS_IP} \
        -e bce_whl_url=${bce_whl_url} \
        -e PORT_RANGE="62000:65536" \
        -e no_proxy=${no_proxy} \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e AGILE_PIPELINE_CONF_ID=${AGILE_PIPELINE_CONF_ID} \
        -e AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID} \
        -e AGILE_JOB_BUILD_ID=${AGILE_JOB_BUILD_ID} \
        -e AGILE_PIPELINE_NAME=${AGILE_PIPELINE_NAME} \
        -e AGILE_WORKSPACE=${AGILE_WORKSPACE} \
        -e REPORT_SERVER_PASSWORD=${REPORT_SERVER_PASSWORD} \
        -e Python_version=${Python_version} \
        -e models_list=${models_list} \
        -e models_file=${models_file} \
        -e system=${system} \
        -e step=${step} \
        -e reponame=${reponame} \
        -e timeout=${timeout} \
        -e mode=${mode} \
        -e use_build=${use_build} \
        -e binary_search_flag=${binary_search_flag} \
        -e analysis_case=${analysis_case} \
        -e use_data_cfs=${use_data_cfs} \
        -e plot=${plot} \
        -e c_plus_plus_predict=${c_plus_plus_predict} \
        -e PaddleX=${PaddleX} \
        -e branch=${branch} \
        -e get_repo=${get_repo} \
        -e paddle_whl=${paddle_whl} \
        -e paddle_inference=${paddle_inference} \
        -e TENSORRT_DIR=${TENSORRT_DIR} \
        -e dataset_org=${dataset_org} \
        -e dataset_target=${dataset_target} \
        -e set_cuda=${set_cuda} \
        -e FLAGS_enable_pir_in_executor=${FLAGS_enable_pir_in_executor} \
	-e FLAGS_enable_pir_api=${FLAGS_enable_pir_api} \
        -e GLOG_vmodule=${GLOG_vmodule} \
        -e ENABLE_FALL_BACK=${ENABLE_FALL_BACK} \
        -e MIN_GRAPH_SIZE=${MIN_GRAPH_SIZE} \
        -e FLAGS_enable_pir_api=${FLAGS_enable_pir_api} \
        -e SOT_EXPORT_FLAG=${SOT_EXPORT_FLAG} \
        -e FLAGS_prim_all=${FLAGS_prim_all} \
        -e FLAGS_use_cinn=${FLAGS_use_cinn} \
	-e api_key=${api_key} \
        -e PTS_ENV_VARS=$PTS_ENV_VARS \
        -e FLAGS_pir_subgraph_saving_dir=${FLAGS_pir_subgraph_saving_dir} \
	-e FLAGS_use_auto_growth_v2=${FLAGS_use_auto_growth_v2} \
        -w /workspace \
        ${Image_version}  \
        /bin/bash -c '
        ldconfig;
        if [[ `yum --help` =~ "yum" ]];then
            echo "centos"
            yum install nfs-utils -y > install_nfs 2>&1
            export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH}
            case ${Python_version} in
            36)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
            ;;
            37)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
            ;;
            38)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
            ;;
            39)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
            ;;
            310)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
            ;;
            311)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.11.0/bin/:${PATH}
            ;;
            312)
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH}
            export PATH=/opt/_internal/cpython-3.12.0/bin/:${PATH}
            ;;
            esac
        else
            echo "ubuntu"
            apt-get update > install_update 2>&1
            apt-get install nfs-common -y > install_nfs 2>&1
            case ${Python_version} in
            36)
            mkdir run_env_py36;
            ln -s $(which python3.6) run_env_py36/python;
            ln -s $(which pip3.6) run_env_py36/pip;
            export PATH=$(pwd)/run_env_py36:${PATH};
            ;;
            37)
            mkdir run_env_py37;
            ln -s $(which python3.7) run_env_py37/python;
            ln -s $(which pip3.7) run_env_py37/pip;
            export PATH=$(pwd)/run_env_py37:${PATH};
            ;;
            38)
            mkdir run_env_py38;
            ln -s $(which python3.8) run_env_py38/python;
            ln -s $(which pip3.8) run_env_py38/pip;
            export PATH=$(pwd)/run_env_py38:${PATH};
            ;;
            39)
            mkdir run_env_py39;
            ln -s $(which python3.9) run_env_py39/python;
            ln -s $(which pip3.9) run_env_py39/pip;
            export PATH=$(pwd)/run_env_py39:${PATH};
            ;;
            310)
            mkdir run_env_py310;
            ln -s $(which python3.10) run_env_py310/python;
            ln -s $(which pip3.10) run_env_py310/pip;
            export PATH=$(pwd)/run_env_py310:${PATH};
            ;;
            311)
            mkdir run_env_py311;
            ln -s $(which python3.11) run_env_py311/python;
            ln -s $(which pip3.11) run_env_py311/pip;
            export PATH=$(pwd)/run_env_py311:${PATH};
            ;;
            312)
            mkdir run_env_py312;
            ln -s $(which python3.12) run_env_py312/python;
            ln -s $(which pip3.12) run_env_py312/pip;
            export PATH=$(pwd)/run_env_py312:${PATH};
            ;;
            esac
        fi
        #挂载数据, 地址特定为mount_path
        export mount_path="/workspace/MT_data/${reponame}"
        if [[ -d ${mount_path} ]];then
            mv ${mount_path} ${mount_path}_back
            mkdir -p ${mount_path}
        else
            mkdir -p ${mount_path}
        fi
        mount -t nfs4 -o minorversion=1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport ${CFS_IP}:/${reponame} ${mount_path}
        ls ${mount_path}
        echo "@@@mount_path: ${mount_path}"

	echo "@@@FLAGS_enable_pir_in_executor: ${FLAGS_enable_pir_in_executor}"
        # FLAGS_enable_pir_in_executor bug
        if [ $FLAGS_enable_pir_in_executor == 0 ];then
        echo "@@@FLAGS_enable_pir_in_executor: ${FLAGS_enable_pir_in_executor}"
        unset FLAGS_enable_pir_in_executor
        fi
    echo "@@@ENABLE_FALL_BACK: ${ENABLE_FALL_BACK}"
        # ENABLE_FALL_BACK install
        if [ $ENABLE_FALL_BACK == True ];then
        echo "@@@ENABLE_FALL_BACK: ${ENABLE_FALL_BACK}"
        echo "@@@MIN_GRAPH_SIZE: ${MIN_GRAPH_SIZE}"
        echo "@@@FLAGS_pir_subgraph_saving_dir: ${FLAGS_pir_subgraph_saving_dir}"
        ehco "@@@export FLAGS_enable_pir_api ${export FLAGS_enable_pir_api}"
        ehco "@@@export SOT_EXPORT_FLAG ${SOT_EXPORT_FLAG}"
        ehco "@@@export GLOG_vmodule ${GLOG_vmodule}"
        set -x
        # Flag
        export STRICT_MODE=0
        export SOT_LOG_LEVEL=2
        export COST_MODEL=False
        python -m pip install git+https://github.com/PaddlePaddle/PaddleSOT@develop
        fi

        nvidia-smi;
        python -c "import sys; print(sys.version_info[:])";
        git --version;
        python -m pip install --user -U pip  -i https://pypi.tuna.tsinghua.edu.cn/simple #升级pip
        python -m pip install --user -U -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple #预先安装依赖包
        python -m pip install --ignore-installed six -i https://pypi.tuna.tsinghua.edu.cn/simple
        python -m pip uninstall -y opencv-python
        python -m pip config list
        python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None} --set_cuda=${set_cuda:-0,1} --timeout=${timeout:-3600} --binary_search_flag=${binary_search_flag:-False} --use_data_cfs=${use_data_cfs:-False} --plot=${plot:-False} --c_plus_plus_predict=${c_plus_plus_predict:-False} --paddle_inference=${paddle_inference:-None} --TENSORRT_DIR=${TENSORRT_DIR:-None} --PaddleX=${PaddleX:-None}
    ' &
    wait $!
    docker start ce_${AGILE_PIPELINE_NAME}_${AGILE_JOB_BUILD_ID}
    docker attach ce_${AGILE_PIPELINE_NAME}_${AGILE_JOB_BUILD_ID}
    exit $?
else
    ldconfig;
    #额外的变量, PORT_RANGE是出现IP_ANY:36986端口占用报错暂时屏蔽一些,221108新出现60636被占用
    export PORT_RANGE=62000:65536
    if [[ `yum --help` =~ "yum" ]];then
        echo "centos"
        export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH}
        case ${Python_version} in
        36)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
        ;;
        37)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
        ;;
        38)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
        ;;
        39)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
        ;;
        310)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
        ;;
        311)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.11.0/bin/:${PATH}
        ;;
        312)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.12.0/bin/:${PATH}
        ;;
        esac
    else
        echo "ubuntu"
        case ${Python_version} in
        36)
        mkdir run_env_py36;
        ln -s $(which python3.6) run_env_py36/python;
        ln -s $(which pip3.6) run_env_py36/pip;
        export PATH=$(pwd)/run_env_py36:${PATH};
        ;;
        37)
        mkdir run_env_py37;
        ln -s $(which python3.7) run_env_py37/python;
        ln -s $(which pip3.7) run_env_py37/pip;
        export PATH=$(pwd)/run_env_py37:${PATH};
        ;;
        38)
        mkdir run_env_py38;
        ln -s $(which python3.8) run_env_py38/python;
        ln -s $(which pip3.8) run_env_py38/pip;
        export PATH=$(pwd)/run_env_py38:${PATH};
        ;;
        39)
        mkdir run_env_py39;
        ln -s $(which python3.9) run_env_py39/python;
        ln -s $(which pip3.9) run_env_py39/pip;
        export PATH=$(pwd)/run_env_py39:${PATH};
        ;;
        310)
        mkdir run_env_py310;
        ln -s $(which python3.10) run_env_py310/python;
        ln -s $(which pip3.10) run_env_py310/pip;
        export PATH=$(pwd)/run_env_py310:${PATH};
        ;;
        311)
        mkdir run_env_py311;
        ln -s $(which python3.11) run_env_py311/python;
        ln -s $(which pip3.11) run_env_py311/pip;
        export PATH=$(pwd)/run_env_py311:${PATH};
        ;;
        312)
        mkdir run_env_py312;
        ln -s $(which python3.12) run_env_py312/python;
        ln -s $(which pip3.12) run_env_py312/pip;
        export PATH=$(pwd)/run_env_py312:${PATH};
        ;;
        esac
    fi

    #集群环境已在节点挂载，任务中配置挂载卷，再需定义mount_path即可
    export mount_path="/home/paddleqa/cfs/${reponame}"
    ls ${mount_path}
    echo "@@@mount_path: ${mount_path}"

    echo "@@@FLAGS_enable_pir_in_executor: ${FLAGS_enable_pir_in_executor}"
    # FLAGS_enable_pir_in_executor bug
    if [ $FLAGS_enable_pir_in_executor == 0 ];then
    echo "@@@FLAGS_enable_pir_in_executor: ${FLAGS_enable_pir_in_executor}"
    unset FLAGS_enable_pir_in_executor
    fi
    echo "@@@ENABLE_FALL_BACK: ${ENABLE_FALL_BACK}"
        # ENABLE_FALL_BACK install
        if [ $ENABLE_FALL_BACK == True ];then
        echo "@@@ENABLE_FALL_BACK: ${ENABLE_FALL_BACK}"
        echo "@@@MIN_GRAPH_SIZE: ${MIN_GRAPH_SIZE}"
        echo "@@@FLAGS_pir_subgraph_saving_dir: ${FLAGS_pir_subgraph_saving_dir}"
        echo "@@@export FLAGS_enable_pir_api: ${export FLAGS_enable_pir_api}"
        echo "@@@export SOT_EXPORT_FLAG: ${SOT_EXPORT_FLAG}"
        echo "@@@export GLOG_vmodule: ${GLOG_vmodule}"

        set -x
        # Flag
        export STRICT_MODE=0
        export SOT_LOG_LEVEL=2
        export COST_MODEL=False
        python -m pip install git+https://github.com/PaddlePaddle/PaddleSOT@develop
        fi

    nvidia-smi;
    python -c "import sys; print(sys.version_info[:])";
    git --version;
    python -m pip install --user -U pip  -i https://mirror.baidu.com/pypi/simple #升级pip
    python -m pip install --user -U -r requirements.txt  -i https://mirror.baidu.com/pypi/simple #预先安装依赖包
    python -m pip install --ignore-installed six -i https://mirror.baidu.com/pypi/simple
    python -m pip uninstall -y opencv-python
    python -m pip config list
    python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None} --set_cuda=${set_cuda:-0,1} --timeout=${timeout:-3600} --binary_search_flag=${binary_search_flag:-False} --use_data_cfs=${use_data_cfs:-False} --plot=${plot:-False} --c_plus_plus_predict=${c_plus_plus_predict:-False} --paddle_inference=${paddle_inference:-None} --TENSORRT_DIR=${TENSORRT_DIR:-None} --PaddleX=${PaddleX:-None}
fi
