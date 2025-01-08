import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdxl", filename="pytorch_lora_weights.safetensors", local_dir="./checkpoints")

import os
os.environ["USE_PEFT_BACKEND"] = "True"
from ppdiffusers import LCMScheduler

lora_state_dict = './checkpoints/pytorch_lora_weights.safetensors'
base_model_path = 'wangqixun/YamerMIX_v8'
pipe.scheduler=LCMScheduler.from_pretrained(base_model_path,
                    subfolder="scheduler",
                    from_hf_hub=True,
                    from_diffusers=True)
pipe.load_lora_weights(lora_state_dict, from_diffusers=True)
pipe.fuse_lora()

num_inference_steps = 10
guidance_scale = 0
