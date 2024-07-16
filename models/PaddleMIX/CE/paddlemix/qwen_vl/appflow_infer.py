# qwen vl appflow inference
import paddle
from paddlemix.appflow import Appflow
paddle.seed(1234)
task = Appflow(app="image2text_generation",
                   models=["qwen-vl/qwen-vl-chat-7b"])
image= "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
prompt = "这是什么？"
result = task(image=image,prompt=prompt)

print(result["result"])

prompt2 = "框出图中公交车的位置"
result = task(prompt=prompt2)
print(result["result"])
