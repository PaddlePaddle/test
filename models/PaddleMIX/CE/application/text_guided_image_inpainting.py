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
text_guided_image_inpainting
"""
import paddle
from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url)
mask_image = load_image(mask_url)
paddle.seed(1024)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

app = Appflow(app='inpainting',models=['stabilityai/stable-diffusion-2-inpainting'])
image = app(inpaint_prompt=prompt,image=image,seg_masks=mask_image)['result']

image.save("a_yellow_cat.png")