
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl --user
# 由于YOLO-World实现依赖PaddleYOLO, 先将PaddleYOLO clone至third_party目录下
mkdir third_party
git clone https://github.com/PaddlePaddle/PaddleYOLO.git third_party/PaddleYOLO

# 安装paddledet
pip install -e third_party/PaddleYOLO

# 安装其他所需的依赖
pip install -e .

# 创建目录存放预训练模型
mkdir pretrain
wget https://bj.bcebos.com/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pdparams
