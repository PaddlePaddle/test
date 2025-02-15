# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
audio2caption_generation
"""
# audio2caption -- Audio to caption converter

from paddlemix.appflow import Appflow
import paddle
paddle.seed(1024)
task = Appflow(app="audio2caption", models=[
               "conformer_u2pp_online_wenetspeech", "THUDM/chatglm-6b"])
audio_file = "./zh.wav"
prompt = (
    "描述这段话：{}."
)
result = task(audio=audio_file, prompt=prompt)['prompt']
print(result)
