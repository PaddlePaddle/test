from ppdiffusers import KandinskyV22CombinedPipeline

output_dir = "kandi22-decoder-pokemon-lora"
pipe = KandinskyV22CombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder")
pipe.unet.load_attn_procs(output_dir)

prompt = 'A robot pokemon, 4k photo'
image = pipe(prompt=prompt).images[0]
image.save("robot_pokemon_lora_decoder_infer.png")
