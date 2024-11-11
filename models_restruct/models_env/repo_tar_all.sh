#!/bin/bash
work_dir=/home/repo_tar_all
if [[ -d "/home/repo_tar_all" ]];then
    echo "already have file /home/repo_tar_all"
    rm -rf /home/repo_tar_all
fi
mkdir -p ${work_dir}
# rm -rf ${work_dir}/*
cd ${work_dir}

# Check repo_name.tar.gz in  BOS
set +xe
unset http_proxy
unset https_proxy
set -xe

export repo_name_all=${repo_name_all:-"PaddleNLP PaddleCustomDevice Paddle PaddleClas PaddleGAN PaddleOCR Paddle3D PaddleSpeech PaddleRec PaddleSlim PaddleDetection PaddleSeg PaddleMIX"}
# Paddle  需要包含 PaddlePaddle  release 字段，需要打包 develop release/2.3 release/2.4
# PaddleClas  需要包含 develop  release 字段，需要打包 develop release/2.3 release/2.4 release/2.5
# PaddleGAN  需要包含 develop  release 字段，需要打包 develop release/2.1
# PaddleOCR: dygraph、release/2.6;
# Paddle3D: develop;
# PaddleSpeech: develop;
# PaddleRec： master
# PaddleSlim： develop、 release/2.4
# PaddleDetection:develop, release2.5
# PaddleSeg:develop,release2.6
# PaddleNLP:develop
# PaddleCustomDevice:develop

function tar_reponame(){
    echo "repo_name is : ${repo_name}"
    line=${line/remotes\//}
    line=${line/origin\//}

    echo "checkout to ${line}"
    git checkout ${line}

    line=${line//\//\-} #使用-链接和github保持一致
    echo "tar branch is ${line}"
    cd ..
    mv ${repo_name} ${repo_name}-${line}
    tar -zcf ${repo_name}-${line}.tar.gz ${repo_name}-${line}
    file_tgz=${repo_name}-${line}.tar.gz
    mv ${repo_name}-${line} ${repo_name}

    if [[ ! -f ${python_name} ]];then
        set +x
        wget -q --no-proxy ${bce_whl_url} --no-check-certificate
        set -x
        tar xf ${tar_name}
    fi
    python3 ${python_name}  ${file_tgz}  "xly-devops/PaddleTest/${repo_name}/"
    echo "upload ${file_tgz} done"

    # 及时删除防止空间打满
    if [[ -f ${file_tgz} ]];then
        echo "remove tmp data : ${file_tgz}"
        rm -rf ${file_tgz}
    fi

    cd ${repo_name}
}


for repo_name in ${repo_name_all}
do
    # Git clone
    if [ -d ${repo_name} ]; then rm -rf ${repo_name}; fi
    echo "start download ${repo_name}"
    git clone https://github.com/PaddlePaddle/${repo_name}.git

    if [[ -d ${repo_name} ]];then
        #打默认分支的包
        tar -zcf ${repo_name}.tar.gz ${repo_name}
        file_tgz=${repo_name}.tar.gz

        if [[ ! -f ${python_name} ]];then
            set +x
            wget -q --no-proxy ${bce_whl_url} --no-check-certificate
            set -x
            tar xf ${tar_name}
        fi
        python3 ${python_name}  ${file_tgz}  "xly-devops/PaddleTest/${repo_name}/"
        echo "upload ${file_tgz} done"

        # 及时删除防止空间打满
        if [[ -f ${file_tgz} ]];then
            echo "remove tmp data : ${file_tgz}"
            rm -rf ${file_tgz}
        fi

        #打满足条件分支的包
        cd ${repo_name}
        git branch -r |while read line
        do
        # Paddle
        if ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/0" ]] \
            && [[ ! $line =~ "release/lite-0.1" ]] \
            && [[ ! $line =~ "release/1" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ! $line =~ "release/2.3-fc-ernie-fix" ]] \
            && [[ ${repo_name} == "Paddle" ]]; then
            tar_reponame
        # PaddleClas
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ! $line =~ "release/2.0-beta" ]] \
            && [[ ! $line =~ "release/2.0-rc1" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ! $line =~ "release/static" ]] \
            && [[ ${repo_name} == "PaddleClas" ]]; then
            tar_reponame
        # PaddleGAN
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/0.1.0" ]]\
            && [[ ! $line =~ "release/2.0-beta" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ${repo_name} == "PaddleGAN" ]]; then
            tar_reponame
        # PaddleOCR
        elif ([[ $line =~ "release" ]] || [[ $line =~ "dygraph" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/1.1" ]] \
            && [[ ! $line =~ "release/2.0-rc1-0" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ! $line =~ "release/2.3" ]] \
            && [[ ! $line =~ "revert-7437-dygraph" ]] \
            && [[ ${repo_name} == "PaddleOCR" ]]; then
            tar_reponame
        # PaddleSpeech
        elif ([[ $line =~ "r1.2" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "r0.1" ]] \
            && [[ ! $line =~ "r0.2" ]] \
            && [[ ! $line =~ "r1.0" ]] \
            && [[ ${repo_name} == "PaddleSpeech" ]]; then
            tar_reponame
        # Paddle3D
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ${repo_name} == "Paddle3D" ]]; then
            tar_reponame
        # PaddleNLP
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]] \
            || [[ $line =~ "refactor-training-loop" ]] || [[ $line =~ "fleety" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ${repo_name} == "PaddleNLP" ]]; then
            tar_reponame
        # PaddleDetection
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/0.1" ]] \
            && [[ ! $line =~ "release/0.2" ]] \
            && [[ ! $line =~ "release/0.3" ]] \
            && [[ ! $line =~ "release/0.4" ]] \
            && [[ ! $line =~ "release/0.5" ]] \
            && [[ ! $line =~ "release/2.0-beta" ]] \
            && [[ ! $line =~ "release/2.0-rc" ]] \
            && [[ ! $line =~ "release/2.0" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ${repo_name} == "PaddleDetection" ]]; then
            tar_reponame
        # PaddleSeg
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/v0.1.0" ]] \
            && [[ ! $line =~ "release/v0.2.0" ]] \
            && [[ ! $line =~ "release/v0.3.0" ]] \
            && [[ ! $line =~ "release/v0.4.0" ]] \
            && [[ ! $line =~ "release/v0.5.0" ]] \
            && [[ ! $line =~ "release/v0.6.0" ]] \
            && [[ ! $line =~ "release/v0.7.0" ]] \
            && [[ ! $line =~ "release/v0.8.0" ]] \
            && [[ ! $line =~ "release/v2.0" ]] \
            && [[ ! $line =~ "release/v2.0.0-rc" ]] \
            && [[ ! $line =~ "release/2.1" ]] \
            && [[ ! $line =~ "release/2.2" ]] \
            && [[ ! $line =~ "release/2.3" ]] \
            && [[ ${repo_name} == "PaddleSeg" ]]; then
            tar_reponame
        # PaddleSlim
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/0.1.0" ]] \
            && [[ ! $line =~ "release/1.0.0" ]] \
            && [[ ! $line =~ "release/1.0.1" ]] \
            && [[ ! $line =~ "release/1.0.2" ]] \
            && [[ ! $line =~ "release/1.1.0" ]] \
            && [[ ! $line =~ "release/1.1.1" ]] \
            && [[ ! $line =~ "release/1.1.2" ]] \
            && [[ ! $line =~ "release/1.2.0" ]] \
            && [[ ! $line =~ "release/1.3.0" ]] \
            && [[ ! $line =~ "release/2.0-alpha" ]] \
            && [[ ! $line =~ "release/2.0.0" ]] \
            && [[ ! $line =~ "release/2.1.0" ]] \
            && [[ ${repo_name} == "PaddleSlim" ]]; then
            tar_reponame
        # PaddleRec
        elif ([[ $line =~ "release" ]] || [[ $line =~ "master" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/1.8.5" ]] \
            && [[ ! $line =~ "release/2.0.0" ]] \
            && [[ ! $line =~ "release/2.1.0" ]] \
            && [[ ${repo_name} == "PaddleRec" ]]; then
            tar_reponame
        # PaddleX
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "release/2" ]] \
            && [[ ${repo_name} == "PaddleX" ]]; then
            tar_reponame
        # PaddleMIX
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ! $line =~ "upgrade_ppdiffusers0193" ]] \
            && [[ ! $line =~ "release/0.1" ]] \
            && [[ ${repo_name} == "PaddleMIX" ]]; then
            tar_reponame
        # PaddleCustomDevice
        elif ([[ $line =~ "release" ]] || [[ $line =~ "develop" ]]) \
            && [[ ! $line =~ "HEAD" ]] \
            && [[ ${repo_name} == "PaddleCustomDevice" ]]; then
            tar_reponame
        else
            echo "${repo_name} not other branch to tar"
        fi
        done
        cd ..
        # 及时删除防止空间打满
        if [[ -d ${repo_name} ]];then
            echo "remove tmp data : ${repo_name}"
            rm -rf ${repo_name}
        fi
    else
        echo "clone ${repo_name} failed"
    fi
done
