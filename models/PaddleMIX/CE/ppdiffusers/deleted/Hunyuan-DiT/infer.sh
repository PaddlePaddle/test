# Prompt Enhancement + Text-to-Image. Paddle mode
python sample_t2i.py --prompt "渔舟唱晚"

# Only Text-to-Image. Paddle mode
python sample_t2i.py --prompt "渔舟唱晚" --no-enhance

# Only Text-to-Image. Flash Attention mode
python sample_t2i.py --infer-mode fa --prompt "渔舟唱晚"

# Generate an image with other image sizes.
python sample_t2i.py --prompt "渔舟唱晚" --image-size 1280 768
