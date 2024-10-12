from ppdiffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import paddle

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs("./lora_dream_outputs")

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("demo.png")