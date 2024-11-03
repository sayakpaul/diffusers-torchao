# AoT compilation for Diffusers

Reference: [https://gist.github.com/zou3519/2f9b97add3eac216ad795397cd942a7c](https://gist.github.com/zou3519/2f9b97add3eac216ad795397cd942a7c) 

***TL;DR***: We can compile a PyTorch model ahead-of-time (AoT) and obtain faster inference latency than just-in-time (JiT) compilation. The numbers below are for the DiT of Flux.1 Dev: 

```bash
JiT compilation: 1.776 seconds
AoT compilation: 0.421 seconds
```

## Introduction

In production use cases, users would want to compile their models ahead-of-time (AoT) and serialize the compiled binary for later use. This way they don’t have to rely on caching or compiling just-in-time. This also helps reduce framework overheads.

For `diffusers` users, this could be quite beneficial as we can then:

- AoT compile the most computationally heavy module for the most widely used dimensions (batch_size, height, width)
- Use them during serving depending on the input payload

This, of course, has a restriction because we’re now limited to using the batch size, height, and width we AoT compiled for. Another option is to compile with `dynamic=True`, but this doesn’t provide the speed benefits as much. 

On the other hand, AoT compilation could really be beneficial for use cases, where we know the shapes we will be dealing with beforehand.

## Benchmarking JiT `torch.compile()`

If we just compile the transformer of Flux.1 Dev, on an A100, we obtain 1.776 seconds for a batch size of 1 and resolution of 1024x1024. 

<details>
<summary>Code</summary>
    
```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch.utils.benchmark as benchmark

def prepare_latents(batch_size, height, width, num_channels_latents=1):
    vae_scale_factor = 16
    height = 2 * (int(height) // vae_scale_factor)
    width = 2 * (int(width) // vae_scale_factor)
    shape = (batch_size, num_channels_latents, height, width)
    pre_hidden_states = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    hidden_states = FluxPipeline._pack_latents(
        pre_hidden_states, batch_size, num_channels_latents, height, width
    )
    return hidden_states

def get_example_inputs(batch_size, height, width, num_channels_latents=1):
    hidden_states = prepare_latents(batch_size, height, width, num_channels_latents)
    num_img_sequences = hidden_states.shape[1]
    example_inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": torch.randn(batch_size, 512, 4096, dtype=torch.bfloat16, device="cuda"),
        "pooled_projections": torch.randn(batch_size, 768, dtype=torch.bfloat16, device="cuda"),
        "timestep": torch.tensor([1.0], device="cuda").expand(batch_size),
        "img_ids": torch.randn(num_img_sequences, 3, dtype=torch.bfloat16, device="cuda"),
        "txt_ids": torch.randn(512, 3, dtype=torch.bfloat16, device="cuda"),
        "guidance": torch.tensor([3.5],  device="cuda").expand(batch_size),
        "return_dict": False,
    }
    return example_inputs

def load_model():
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    return model

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

@torch.no_grad()
def f(model, **kwargs):
    return model(**kwargs)

model = load_model()
num_channels_latents = model.config.in_channels // 4 
inputs = get_example_inputs(batch_size=1, height=1024, width=1024, num_channels_latents=num_channels_latents)

for _ in range(5):
    _ = f(model, **inputs)

model = torch.compile(model, mode="max-autotune")

time = benchmark_fn(f, model, **inputs)
print(time) # 1.776 seconds on A100.
```

</details>    

## AoT `torch.compile()`

First, we obtain the AoT compiled `.so` file:

<details>
</summary>Code</summary>
    
```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
import torch.utils.benchmark as benchmark
from functools import partial

def get_example_inputs():
    example_inputs = {
        "hidden_states": torch.load("latents.pt", map_location="cuda"),
        "encoder_hidden_states": torch.load("prompt_embeds.pt", map_location="cuda"),
        "pooled_projections": torch.load("pooled_prompt_embeds.pt", map_location="cuda"),
        "timestep": torch.load("timestep.pt", map_location="cuda") / 1000,
        "img_ids": torch.load("latent_image_ids.pt", map_location="cuda"),
        "txt_ids": torch.load("text_ids.pt", map_location="cuda"),
        "guidance": torch.load("guidance.pt", map_location="cuda"),
        "joint_attention_kwargs": None, 
        "return_dict": False
    }
    return example_inputs

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"

def load_model():
    model = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda")
    return model

def aot_compile(name, fn, **sample_kwargs):
    path = f"{name}.so"
    print(f"{path=}")
    options = {
        "aot_inductor.output_path": path,
        "max_autotune": True,
        "triton.cudagraphs": True,
    }

    torch._export.aot_compile(
        fn,
        (),
        sample_kwargs,
        options=options,
        disable_constraint_solver=True,
    )
    return path 

def aot_load(path):
    return torch._export.aot_load(path, "cuda")

@torch.no_grad()
def f(model, **kwargs):
    return model(**kwargs)

model = load_model()
num_channels_latents = model.config.in_channels // 4 

inputs = get_example_inputs()
path = aot_compile("bs_1_1024", partial(f, model=model), **inputs)

compiled_func_1 = aot_load(path)
print(f"{compiled_func_1(**inputs)[0].shape=}")

for _ in range(5):
    _ = compiled_func_1(**inputs)[0]

time = benchmark_fn(f, compiled_func_1, **inputs)
print(time) # 0.421 seconds on an A100.
```

</details>    

Here, we’re obtaining the compiled graph for a batch size of 1 and a resolution of 1024x1024. The example inputs for the AoT compilation were manually obtained from `pipeline_flux.py` ([implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py)) by serializing the inputs (inputs that go to `self.transformer` [here](https://github.com/huggingface/diffusers/blob/fddbab79932eedf1a78041ef38c47df80ab84c90/src/diffusers/pipelines/flux/pipeline_flux.py#L732)) to have reasonable values rather than random values. 

Then, run the full pipeline:

```python
import torch 
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=None,
    torch_dtype=torch.bfloat16,
).to("cuda")
pipeline.transformer = torch._export.aot_load("/raid/.cache/bs_1_1024.so", "cuda")

image = pipeline(
    "cute dog", guidance_scale=3.5, max_sequence_length=512, num_inference_steps=50
).images[0]
image.save("aot_compiled.png")
```

Note that running the above requires the following patch:

```diff
diff --git a/src/diffusers/pipelines/flux/pipeline_flux.py b/src/diffusers/pipelines/flux/pipeline_flux.py
index bb214885d..78d04d51a 100644
--- a/src/diffusers/pipelines/flux/pipeline_flux.py
+++ b/src/diffusers/pipelines/flux/pipeline_flux.py
@@ -667,7 +667,7 @@ class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
         )
 
         # 4. Prepare latent variables
-        num_channels_latents = self.transformer.config.in_channels // 4
+        num_channels_latents = self.transformer.config.in_channels // 4 if isinstance(self.transformer, torch.nn.Module) else 16
         latents, latent_image_ids = self.prepare_latents(
             batch_size * num_images_per_prompt,
             num_channels_latents,
@@ -701,7 +701,7 @@ class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
         self._num_timesteps = len(timesteps)
 
         # handle guidance
-        if self.transformer.config.guidance_embeds:
+        if (isinstance(self.transformer, torch.nn.Module) and self.transformer.config.guidance_embeds) or isinstance(self.transformer, Callable):
             guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
             guidance = guidance.expand(latents.shape[0])
         else:

```

## Known limitations

1. Doesn’t work with `load_lora_weights()`. You must fuse the LoRA weights and unload them before performing AoT compilation. 
2. Doesn’t work with `enable_model_cpu_offload()` yet. 

## Quantization + AoT compilation?

[https://gist.github.com/sayakpaul/de0eeeb6d08ba30a37dcf0bc9dacc5c5](https://gist.github.com/sayakpaul/de0eeeb6d08ba30a37dcf0bc9dacc5c5)