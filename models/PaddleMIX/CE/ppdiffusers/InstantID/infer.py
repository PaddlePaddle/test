import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdxl", filename="pytorch_lora_weights.safetensors", local_dir="./checkpoints")

import paddle
import cv2
import os
os.environ["USE_PEFT_BACKEND"] = "True"
import numpy as np
from PIL import Image
from ppdiffusers import ControlNetModel, AutoencoderKL
from ppdiffusers.utils import load_image
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
face_adapter = f'./InstantID/checkpoints/ip-adapter.bin'
controlnet_path = f'./InstantID/checkpoints/ControlNetModel'
controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                             paddle_dtype=paddle.float16,
                                             use_safetensors=True,
                                             from_hf_hub=True,
                                             from_diffusers=True)

base_model_path = "wangqixun/YamerMIX_v8"

vae = AutoencoderKL.from_pretrained(base_model_path, from_diffusers=True, from_hf_hub=True, subfolder="vae")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(base_model_path,
                                                          controlnet=controlnet,
                                                          paddle_dtype=paddle.float16,
                                                          from_diffusers=True,
                                                          from_hf_hub=True,
                                                          low_cpu_mem_usage=True)
pipe.vae = vae
pipe.load_ip_adapter_instantid(face_adapter,
                               weight_name=os.path.basename("face_adapter"),
                               from_diffusers=True)