from ppdiffusers import StableDiffusionXLPipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
import paddle

unet_path = "your-checkpoint/unet"

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
unet = UNet2DConditionModel.from_pretrained(unet_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")