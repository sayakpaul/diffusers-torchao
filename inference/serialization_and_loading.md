# Serializing a quantized model

TorchAO seamlessly integrates quantization into the PyTorch ecosystem. While performing inference with lower precision data types is a significant advantage, the ability to serialize models in a quantized format to reduce disk space usage has been a challenging requirement for many existing frameworks. TorchAO addresses this need effectively!

Following are two examples that demonstrate end-to-end serialization and de-serialization of quantized checkpoints:

- [Flux](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47)
- [CogVideoX](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897#file-quantized_serialize_and_unserialized-py)

Before proceeding, please install `huggingface_hub` from the source: `pip install git+https://github.com/huggingface/huggingface_hub`. 

### Flux

```python
from diffusers import FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
import torch

ckpt_id = "black-forest-labs/FLUX.1-schnell"

transformer = FluxTransformer2DModel.from_pretrained(
    ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
quantize_(transformer, int8_weight_only())
output_dir = "./flux-schnell-int8wo"
transformer.save_pretrained(output_dir, safe_serialization=False)

# Push to the Hub optionally.
# save_to = "sayakpaul/flux-schnell-int8wo"
# transformer.push_to_hub(save_to, safe_serialization=False)
```

### CogVideoX

```python
import torch
from diffusers import CogVideoXTransformer3DModel, CogVideoXPipeline
from torchao.quantization import quantize_, int8_weight_only

# Either "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
model_id = "THUDM/CogVideoX-5b"

# Quantize and save the transformer
transformer = CogVideoXTransformer3DModel.from_pretrained(
  model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
quantize_(transformer, int8_weight_only())

transformer.save_pretrained("cog-5b-transformer-int8", safe_serialization=False)
```

# Deserializing and loading a quantized model

### Flux

```python
import torch
from diffusers import FluxTransformer2DModel, DiffusionPipeline

dtype, device = torch.bfloat16, "cuda"
ckpt_id = "black-forest-labs/FLUX.1-schnell"

model = FluxTransformer2DModel.from_pretrained(
    "sayakpaul/flux.1-schell-int8wo-improved", torch_dtype=dtype, use_safetensors=False
)
pipeline = DiffusionPipeline.from_pretrained(ckpt_id, transformer=model, torch_dtype=dtype).to("cuda")
image = pipeline(
	"cat", guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256
).images[0]
image.save("flux_schnell_int8.png")
```

You can refer to [this gist](https://gist.github.com/sayakpaul/1f543120a3c4d6ffebb682bbc80f9805) that also benchmarks the time and memory of the loaded quantized model. From the Flux.1-Dev benchmarks, we already know that `int8wo` latency is more than the non-quantized variant. The gain is in reduced memory consumption. Below are the numbers obtained on an H100:

```
no quant
time='0.660' seconds.
memory='31.451' GB.

int8wo
time='0.735' seconds.
memory='20.517' GB.
```

### CogVideoX

```python
import torch
from diffusers import CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video

# Load quantized model
transformer = CogVideoXTransformer3DModel.from_pretrained(
    "cog-5b-transformer-int8", torch_dtype=torch.bfloat16, use_safetensors=False
)

# Create pipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Run inference
prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
video = pipe(
    prompt=prompt,
    guidance_scale=6,
    use_dynamic_cfg=True,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(3047),  # https://arxiv.org/abs/2109.08203
).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

> [!IMPORTANT]  
> Currently, saving quantized models in safetensors format is not supported, and one must use the pytorch tensor format. In the near future, serializing in safetensors will also be possible thanks to the efforts of [Jerry Zhang](https://github.com/jerryzh168) in [this PR](https://github.com/huggingface/safetensors/pull/516).

## Ahead-of-Time (AoT) compilation and serialization

Check out the guide here[./aot_serialization.md].