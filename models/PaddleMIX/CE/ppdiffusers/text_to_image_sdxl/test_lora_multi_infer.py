# multi image
from ppdiffusers import StableDiffusionXLPipeline
import paddle
import os

dir_name = "./sd-pokemon-model-lora-sdxl/"
for file_name in sorted(os.listdir(dir_name)):
    print(file_name)
    model_path = os.path.join(dir_name, file_name)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
    pipe.load_lora_weights(model_path)

    prompt = "A pokemon with green eyes and red legs."
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("pokemon_" + file_name + ".png")