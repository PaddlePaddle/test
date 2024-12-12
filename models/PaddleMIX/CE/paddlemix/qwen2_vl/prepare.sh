mix_path=${root_path}/PaddleMIX


# 数据集下载
cd ${mix_path}
if [ -e "playground.tar" ]; then
    tar -xf playground.tar
    echo "playground.tar文件存在，已解压"
else
    wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar
    tar -xf playground.tar
    echo "playground不存在，已下载解压"
fi

cd playground
if [ -e "opensource_json.tar"]; then
    tar xf opensource_json.tar
    echo "opensource_json.tar文件存在，已解压"
else
    wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
    tar xf opensource_json.tar
    echo "opensource_json.tar不存在，已下载解压"
fi


