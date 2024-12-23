import paddle
from ppdiffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, paddle_dtype=paddle.float16)

# 我们可以开启 xformers
# pipe.enable_xformers_memory_efficient_attention()

# Apply ToMe with a 50% merging ratio
pipe.apply_tome(ratio=0.5) # Can also use pipe.unet in place of pipe here

generator = paddle.Generator().manual_seed(0)
image = pipe("a photo of an astronaut riding a horse on mars", generator=generator).images[0]
image.save("astronaut.png")