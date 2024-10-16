from ppdiffusers import StableDiffusion3Pipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
import paddle

transformer_path = "trained-sd3/checkpoint-10/transformer"

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", paddle_dtype=paddle.float16
)
transformer = SD3Transformer2DModel.from_pretrained(transformer_path)

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog_dreambooth_finetune.png")