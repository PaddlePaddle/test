from ppdiffusers import KandinskyV22CombinedPipeline, DiffusionPipeline

output_dir = "kandi2-prior-pokemon-model"
pipe_prior = DiffusionPipeline.from_pretrained(output_dir)
prior_components = {"prior_" + k: v for k, v in pipe_prior.components.items()}
pipe = KandinskyV22CombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", **prior_components)

prompt = 'A robot pokemon, 4k photo'
images = pipe(prompt=prompt).images
images[0].save("robot-pokemon_infer_prior.png")
