
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl
# 由于YOLO-World实现依赖PaddleYOLO, 先将PaddleYOLO clone至third_party目录下
mkdir third_party
git clone https://github.com/PaddlePaddle/PaddleYOLO.git third_party/PaddleYOLO

# 安装paddledet
pip install -e third_party/PaddleYOLO

# 安装其他所需的依赖
pip install -e .

# 创建目录存放预训练模型
wget https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg
mkdir pretrain
cd pretrain
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/yoloworldv2/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain-8698fbfa.pdparams
