#下载训练数据集
if [ -d "./ScienceQA" ]; then
    echo "ScienceQA目录存在"
else
    if [ -e "ScienceQA.tar" ]; then
        tar -xvf ScienceQA.tar
    else
        wget https://bj.bcebos.com/v1/paddlenlp/datasets/examples/ScienceQA.tar
        tar -xvf ScienceQA.tar
        echo "文件和目录都不存在, 下载数据集成功"
    fi
fi

#准备数据
rm -rf 000000004505.jpg
wget https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg

pip list