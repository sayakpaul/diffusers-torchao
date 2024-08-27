# Serializing a quantized model

TorchAO seamlessly integrates quantization into the PyTorch ecosystem. While performing inference with lower precision data types is a significant advantage, the ability to serialize models in a quantized format to reduce disk space usage has been a challenging requirement for many existing frameworks. TorchAO addresses this need effectively!

Following are two examples that demonstrate end-to-end serialization and de-serialization of quantized checkpoints:

- [Flux](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47)
- [CogVideoX](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897#file-quantized_serialize_and_unserialized-py)

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

torch.save(transformer.state_dict(), "flux_schnell_int8wo.pt")
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

torch.save(transformer.state_dict(), "cog-5b-transformer-int8.pt")
```

# Deserializing and loading a quantized model

### Flux

```python
import torch
from huggingface_hub import hf_hub_download
from diffusers import FluxTransformer2DModel, DiffusionPipeline

dtype, device = torch.bfloat16, "cuda"
ckpt_id = "black-forest-labs/FLUX.1-schnell"

with torch.device("meta"):
    config = FluxTransformer2DModel.load_config(ckpt_id, subfolder="transformer")
    model = FluxTransformer2DModel.from_config(config).to(dtype)

ckpt_path = hf_hub_download(repo_id="sayakpaul/flux.1-schell-int8wo", filename="flux_schnell_int8wo.pt")
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict, assign=True)

pipeline = DiffusionPipeline.from_pretrained(ckpt_id, transformer=model, torch_dtype=dtype).to("cuda")
image = pipeline(
	"cat", guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256
).images[0]
image.save("flux_schnell_int8.png")
```

### CogVideoX

```python
import torch
from diffusers import CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video

# Load quantized model
state_dict = torch.load("cog-5b-transformer-int8.pt")
transformer = CogVideoXTransformer3DModel.from_config(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.load_state_dict(state_dict, assign=True, strict=True)

# Create pipeline
pipe = CogVideoXPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")

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
