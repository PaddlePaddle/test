import paddle

from ppdiffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from ppdiffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "your-stable-video-diffusion-img2vid-model-path-or-id",
    paddle_dtype=paddle.float16
)

# Load the conditioning image
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=")
image = load_image("rocket.png")
image = image.resize((1024, 576))

generator = paddle.Generator().manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)