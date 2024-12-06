# 下载数据集
if [ -e "ScienceQA.tar" ]; then
    tar -xf ScienceQA.tar
    echo "文件存在"
else
    wget https://bj.bcebos.com/v1/paddlenlp/datasets/examples/ScienceQA.tar
    tar -xf ScienceQA.tar
    echo "文件不存在，已下载解压"
fi

