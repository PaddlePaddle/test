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
infer
"""
from ppdiffusers import StableDiffusionControlNetPipeline, ControlNetModel
from ppdiffusers.utils import load_image
controlnet = ControlNetModel.from_pretrained(
    "./sd15_control/checkpoint-100/controlnet")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None)
canny_edged_image = load_image(
    "https://user-images.githubusercontent.com/50394665/221844474-fd539851-7649-470e-bded-4d174271cc7f.png")
img = pipe(prompt="pale golden rod circle with old lace background",
           image=canny_edged_image, guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")
