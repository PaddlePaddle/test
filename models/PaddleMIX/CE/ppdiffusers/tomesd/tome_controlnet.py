import paddle
from ppdiffusers import ControlNetModel, StableDiffusionControlNetPipeline
from ppdiffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet, paddle_dtype=paddle.float16
)

# Apply ToMe with a 50% merging ratio
pipe.apply_tome(ratio=0.5) # Can also use pipe.unet in place of pipe here

# 我们可以开启 xformers
# pipe.enable_xformers_memory_efficient_attention()
generator = paddle.Generator().manual_seed(0)
prompt = "bird"
image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
)

image = pipe(prompt, image, generator=generator).images[0]

image.save("bird.png")