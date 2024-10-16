from ppdiffusers import StableDiffusionXLPipeline
import paddle

model_path = "./sd-pokemon-model-lora-sdxl"
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
pipe.load_lora_weights(model_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
