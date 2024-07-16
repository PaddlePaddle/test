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
grounded_sam_inpainting
"""
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
import paddle
task = Appflow(app="inpainting",
               models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024","stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
image_pil =  load_image(url)
result = task(image=image_pil,prompt="bus",inpaint_prompt="a yellow van")