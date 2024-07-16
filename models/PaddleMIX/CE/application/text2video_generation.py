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
text2video_generation
"""
from paddlemix.appflow import Appflow
import imageio


prompt = "An astronaut riding a horse."

app = Appflow(app='text_to_video_generation',models=['damo-vilab/text-to-video-ms-1.7b'])
video_frames = app(prompt=prompt,num_inference_steps=25)['result']

imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.gif", video_frames,duration=8)
