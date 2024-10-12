import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DDIMScheduler, DiTPipeline

dtype = paddle.float32
pipe = DiTPipeline.from_pretrained("./DiT_XL_2_256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)

set_seed(42)
generator = paddle.Generator().manual_seed(0)
image = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator).images[0]
image.save("result_DiT_golden_retriever.png")