# Serializing a quantized model

TorchAO seamlessly integrates quantization into the PyTorch ecosystem. While performing inference with lower precision data types is a significant advantage, the ability to serialize models in a quantized format to reduce disk space usage has been a challenging requirement for many existing frameworks. TorchAO addresses this need effectively!

Following are two examples that demonstrate end-to-end serialization and de-serialization of quantized checkpoints:

- [PixArt-Sigma](<TODO: add link to gist here>)
- [CogVideoX-5b](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897#file-quantized_serialize_and_unserialized-py)

### PixArt-Sigma

```python
# TODO
```

### CogVideoX-5B

```python
import torch
from diffusers import CogVideoXTransformer3DModel, CogVideoXPipeline
from torchao.quantization import quantize_, int8_weight_only

# Either "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
model_id = "THUDM/CogVideoX-5b"

# Quantize and save the transformer
transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
quantize_(transformer, int8_weight_only())

torch.save(transformer.state_dict(), "cog-5b-transformer-int8.pt")
```

# Deserializing and loading a quantized model

### PixArt-Sigma

```python
# TODO
```

### CogVideoX-5B

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
