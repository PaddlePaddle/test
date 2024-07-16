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
grounded_sam_chatglm
"""
import paddle
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
task = Appflow(app="inpainting",
               models=["THUDM/chatglm-6b","GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024","stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
image_pil =  load_image(url)
inpaint_prompt = "bus is changed to A school bus parked on the roadside"
prompt = "Given caption,extract the main object to be replaced and marked it as 'main_object'," \
         + "Extract the remaining part as 'other prompt', " \
         + "Return main_object, other prompt in English" \
         + "Given caption: {}.".format(inpaint_prompt)
result = task(image=image_pil,prompt=prompt)
