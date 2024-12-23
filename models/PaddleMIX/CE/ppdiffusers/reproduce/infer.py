from pathlib import Path
import paddle
from ppdiffusers import StableDiffusionAttendAndExcitePipeline, PNDMScheduler


scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    )

seed = 123
prompt = "A playful kitten chasing a butterfly in a wildflower meadow"
token_indices = [3,6,10]

generator = paddle.Generator().manual_seed(seed)
image = pipe(
    prompt=prompt,
    token_indices=token_indices,
    generator=generator,
).images[0]

# save
output_dir = Path("output_pd")
prompt_output_path = output_dir / prompt
prompt_output_path.mkdir(exist_ok=True, parents=True)
image.save(prompt_output_path / f'{seed}.png')
