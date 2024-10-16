from ppdiffusers import StableDiffusionXLPipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
import paddle
import os

dir_name = "sdxl-pokemon-model"
for file_name in sorted(os.listdir(dir_name)):
    print(file_name)
    unet_path = os.path.join(dir_name, file_name)

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet")

    prompt = "A pokemon with green eyes and red legs."
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("sdxl_train_pokemon_" + file_name + ".png")