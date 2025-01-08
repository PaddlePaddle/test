from ppdiffusers import KandinskyV22CombinedPipeline

output_dir = "./kandi2-decoder-pokemon-model"
pipe = KandinskyV22CombinedPipeline.from_pretrained(output_dir)

prompt = 'A robot pokemon, 4k photo'
images = pipe(prompt=prompt).images
images[0].save("robot-pokemon_infer_decoder.png")
