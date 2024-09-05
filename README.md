# diffusers-torchao

**Optimize image and video generation with [`diffusers`](https://github.com/huggingface/diffusers), [`torchao`](https://github.com/pytorch/ao), combining `torch.compile()` 🔥** 

We provide end-to-end inference and experimental training recipes to use `torchao` with `diffusers` in this repo. We demonstrate **XX%** speedup on [Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and **21%** speedup on [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b) when comparing *compiled* quantized models against their standard bf16 counterparts<sup>*</sup>. 

<sub><sup>*</sup>The experiments were run on a single A100, 80 GB GPU.</sub>

No-frills code:

```diff
from diffusers import FluxPipeline
+ from torchao.quantization import autoquant
import torch 

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")
+ pipeline.transformer = autoquant(pipeline.transformer, error_on_unseen=False)
image = pipeline(
    "a dog surfing on moon", guidance_scale=3.5, num_inference_steps=50
).images[0]
```

Throw in `torch.compile()` to make it go brrr:

```diff
+ pipeline.transformer.to(memory_format=torch.channels_last)
+ pipeline.transformer = torch.compile(
+    pipeline.transformer, mode="max-autotune", fullgraph=True
+)
```

This, alone, is sufficient to cut down inference time for Flux.1-Dev from X seconds to Y seconds on an H100. Check out the [`inference`](./inference/) directory for the code.

> [!NOTE]
> Quantizing to a supported datatype and using base precision as fp16 can lead to overflows. The recommended base precision for CogVideoX-2b is fp16 while that of CogVideoX-5b is bf16. If comparisons were to be made in fp16, the speedup gains would be **~23%** and **~32%** respectively.

<h4>Table of contents</h4>

* [Environment](#environment)
* [Benchmarking results](#benchmarking-results)
* [Training with FP8](#training-with-fp8)
* [Serialization and loading quantized models](#serialization-and-loading-quantized-models)
* [Things to keep in mind when benchmarking](#things-to-keep-in-mind-when-benchmarking)
* [Benefitting from `torch.compile()`](#benefitting-from-torchcompile)

## Environment

We conducted all our experiments on a single A100 (80GB) and H100 GPUs. Since we wanted to benefit from `torch.compile()`, we used relatively modern cards here. For older cards, same memory savings (demonstrated more below) can be obtained.

We always default to using the PyTorch nightly, updated `diffusers` and `torchao` codebases. We used CUDA 12.2.

## Benchmarking results

We benchmark two models ([Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)) using different supported quantization datatypes in `torchao`. The results are as follows:

TODO: Find out what the best way of presenting all the information is. Having multiple giant table might be difficult to parse visually.

<details>
<summary> Flux Benchmarks </summary>

TODO(sayak): Flux benchmarks
</details>

<details>
<summary> CogVideoX Benchmarks </summary>

**A100**

|  model_type  |  compile  |  fuse_qkv  |  quantize_vae  |  quantization  |   model_memory |   inference_memory |    time |
|:------------:|:---------:|:----------:|:--------------:|:--------------:|:--------------:|:------------------:|:-------:|
|      5B      |   False   |   False    |     False      |      fp16      |         19.764 |             31.746 | 258.962 |
|      5B      |   False   |    True    |     False      |      fp16      |         21.979 |             33.961 | 257.761 |
|      5B      |   True    |   False    |     False      |      fp16      |         19.763 |             31.742 | 225.998 |
|      5B      |   True    |    True    |     False      |      fp16      |         21.979 |             33.961 | 225.814 |
|      5B      |   False   |   False    |     False      |      bf16      |         19.764 |             31.746 | 243.312 |
|      5B      |   False   |    True    |     False      |      bf16      |         21.979 |              33.96 | 242.519 |
|      5B      |   True    |   False    |     False      |      bf16      |         19.763 |             31.742 | 212.022 |
|      5B      |   True    |    True    |     False      |      bf16      |         21.979 |             33.961 | 211.377 |
|      5B      |   False   |   False    |     False      |     int8wo     |         10.302 |             22.288 | 260.036 |
|      5B      |   False   |    True    |     False      |     int8wo     |         11.414 |             23.396 | 271.627 |
|      5B      |   True    |   False    |     False      |     int8wo     |         10.301 |             22.282 | 205.899 |
|      5B      |   True    |    True    |     False      |     int8wo     |         11.412 |             23.397 | 209.640 |
|      5B      |   False   |   False    |     False      |     int8dq     |           10.3 |             22.287 | 550.239 |
|      5B      |   False   |    True    |     False      |     int8dq     |         11.414 |             23.399 | 530.113 |
|      5B      |   True    |   False    |     False      |     int8dq     |           10.3 |             22.286 | 177.256 |
|      5B      |   True    |    True    |     False      |     int8dq     |         11.414 |             23.399 | 177.666 |
|      5B      |   False   |   False    |     False      |     int4wo     |          6.237 |             18.221 | 1130.86 |
|      5B      |   False   |    True    |     False      |     int4wo     |          6.824 |             18.806 | 1127.56 |
|      5B      |   True    |   False    |     False      |     int4wo     |          6.235 |             18.217 | 1068.31 |
|      5B      |   True    |    True    |     False      |     int4wo     |          6.825 |             18.809 | 1067.26 |
|      5B      |   False   |   False    |     False      |     int4dq     |          11.48 |             23.463 | 340.204 |
|      5B      |   False   |    True    |     False      |     int4dq     |         12.785 |             24.771 | 323.873 |
|      5B      |   True    |   False    |     False      |     int4dq     |          11.48 |             23.466 | 219.393 |
|      5B      |   True    |    True    |     False      |     int4dq     |         12.785 |             24.774 | 218.592 |
|      5B      |   False   |   False    |     False      |      fp6       |          7.902 |             19.886 | 283.478 |
|      5B      |   False   |    True    |     False      |      fp6       |          8.734 |             20.718 | 281.083 |
|      5B      |   True    |   False    |     False      |      fp6       |            7.9 |             19.885 | 205.123 |
|      5B      |   True    |    True    |     False      |      fp6       |          8.734 |             20.719 | 204.564 |
|      5B      |   False   |   False    |     False      |   autoquant    |         19.763 |             24.938 | 540.621 |
|      5B      |   False   |    True    |     False      |   autoquant    |         21.978 |               27.1 | 504.031 |
|      5B      |   True    |   False    |     False      |   autoquant    |         19.763 |              24.73 | 176.794 |
|      5B      |   True    |    True    |     False      |   autoquant    |         21.978 |             26.948 | 177.122 |
|      5B      |   False   |   False    |     False      |    sparsify    |          6.743 |             18.727 | 308.767 |
|      5B      |   False   |    True    |     False      |    sparsify    |          7.439 |             19.433 | 300.013 |
|      2B      |   False   |   False    |     False      |      fp16      |         12.535 |             24.511 | 96.918  |
|      2B      |   False   |    True    |     False      |      fp16      |         13.169 |             25.142 | 96.610  |
|      2B      |   True    |   False    |     False      |      fp16      |         12.524 |             24.498 | 83.938  |
|      2B      |   True    |    True    |     False      |      fp16      |         13.169 |             25.143 | 84.694  |
|      2B      |   False   |   False    |     False      |      bf16      |          12.55 |             24.528 | 93.896  |
|      2B      |   False   |    True    |     False      |      bf16      |         13.194 |             25.171 | 93.396  |
|      2B      |   True    |   False    |     False      |      bf16      |         12.486 |             24.526 | 81.224  |
|      2B      |   True    |    True    |     False      |      bf16      |          13.13 |             25.171 | 81.520  |
|      2B      |   False   |   False    |     False      |      fp6       |          6.125 |             18.164 | 95.684  |
|      2B      |   False   |    True    |     False      |      fp6       |          6.769 |             18.808 | 91.698  |
|      2B      |   True    |   False    |     False      |      fp6       |          6.125 |             18.164 | 72.261  |
|      2B      |   True    |    True    |     False      |      fp6       |          6.767 |             18.808 | 90.585  |
|      2B      |   False   |   False    |     False      |     int8wo     |           6.58 |             18.621 | 102.941 |
|      2B      |   False   |    True    |     False      |     int8wo     |          6.894 |             18.936 | 102.403 |
|      2B      |   True    |   False    |     False      |     int8wo     |          6.577 |             18.618 | 81.389  |
|      2B      |   True    |    True    |     False      |     int8wo     |          6.891 |              18.93 | 83.079  |
|      2B      |   False   |   False    |     False      |     int8dq     |           6.58 |             18.621 | 197.254 |
|      2B      |   False   |    True    |     False      |     int8dq     |          6.894 |             18.936 | 190.125 |
|      2B      |   True    |   False    |     False      |     int8dq     |           6.58 |             18.621 |  75.16  |
|      2B      |   True    |    True    |     False      |     int8dq     |          6.891 |             18.933 | 74.981  |
|      2B      |   False   |   False    |     False      |     int4dq     |          7.344 |             19.385 | 132.155 |
|      2B      |   False   |    True    |     False      |     int4dq     |          7.762 |             19.743 | 122.657 |
|      2B      |   True    |   False    |     False      |     int4dq     |          7.395 |             19.374 | 83.103  |
|      2B      |   True    |    True    |     False      |     int4dq     |          7.762 |             19.741 | 82.642  |
|      2B      |   False   |   False    |     False      |     int4wo     |          4.155 |             16.138 | 363.792 |
|      2B      |   False   |    True    |     False      |     int4wo     |          4.345 |             16.328 | 361.839 |
|      2B      |   True    |   False    |     False      |     int4wo     |          4.155 |             16.139 | 342.817 |
|      2B      |   True    |    True    |     False      |     int4wo     |          4.354 |             16.339 | 341.48  |
|      2B      |   False   |   False    |     False      |   autoquant    |          12.55 |             19.734 | 185.023 |
|      2B      |   False   |    True    |     False      |   autoquant    |         13.194 |             20.319 | 177.602 |
|      2B      |   True    |   False    |     False      |   autoquant    |          12.55 |             19.565 | 75.005  |
|      2B      |   True    |    True    |     False      |   autoquant    |         13.195 |             20.191 | 74.807  |
|      2B      |   False   |   False    |     False      |    sparsify    |          4.445 |             16.431 | 125.59  |
|      2B      |   False   |    True    |     False      |    sparsify    |          4.652 |             16.635 | 121.357 |


**H100**

|  model_type  |  compile  |  fuse_qkv  |  quantize_vae  |  quantization  |   model_memory |   inference_memory |    time |
|:------------:|:---------:|:----------:|:--------------:|:--------------:|---------------:|-------------------:|--------:|
|      5B      |   False   |    True    |     False      |      fp16      |         21.978 |             33.988 | 113.945 |
|      5B      |   True    |    True    |     False      |      fp16      |         21.979 |              33.99 | 87.155  |
|      5B      |   False   |    True    |     False      |      bf16      |         21.979 |             33.988 | 112.398 |
|      5B      |   True    |    True    |     False      |      bf16      |         21.979 |             33.987 | 87.455  |
|      5B      |   False   |    True    |     False      |      fp8       |         11.374 |             23.383 | 113.167 |
|      5B      |   True    |    True    |     False      |      fp8       |         11.374 |             23.383 | 75.255  |
|      5B      |   False   |    True    |     False      |     int8wo     |         11.414 |             23.422 | 123.144 |
|      5B      |   True    |    True    |     False      |     int8wo     |         11.414 |             23.423 | 87.026  |
|      5B      |   True    |    True    |     False      |     int8dq     |         11.412 |             59.355 | 78.945  |
|      5B      |   False   |    True    |     False      |     int4dq     |         12.785 |             24.793 | 151.242 |
|      5B      |   True    |    True    |     False      |     int4dq     |         12.785 |             24.795 | 87.403  |
|      5B      |   False   |    True    |     False      |     int4wo     |          6.824 |             18.829 | 667.125 |

</details>

Through visual inspection of various outputs, we identified that the best results were achieved with int8 weight-only quantization, int8 dynamic quantization, fp8 (currently supported only on Hopper architecture), and autoquant. While the outputs sometimes differed visually from their standard fp16/bf16 counterparts, they maintained the expected quality. Additionally, we observed that int4 dynamic quantization generally produced satisfactory results in most cases, but showed greater deviation in structure, color, composition and motion.

> [!NOTE]
> From our testing and feedback from various folks that tried out torchao quantization after the release of CogVideoX, it was found that Ampere and above architectures had the best support for quantization dtypes. For other architectures such as Turing or Volta, quantizing the models did not help save memory or the inference errored out. It was particularly pointed out to be erroneous with the Apple `mps` backend. Support for other architectures will only get better with time.

### CogVideoX memory savings

- From the table, it can be seen that the memory required to load the standard bf16 model into memory is about **19.7 GB**, and to run inference is about **31.7 GB**. To keep the quality on par, let's quantize using int8 weight-only. This requires about **10.3 GB** to load the memory in model, and **22.2 GB** to run inference: 
<details>
<summary>Code</summary>

```python3
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only

model_id = "THUDM/CogVideoX-5b"

text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
quantize_(text_encoder, int8_weight_only())

transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
quantize_(transformer, int8_weight_only())

vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
quantize_(vae, int8_weight_only())

# Create pipeline and run inference
pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
video = pipe(prompt=prompt, num_inference_steps=1).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

</details>

- Let's enable CPU offloading for models as described in [diffusers-specific optimizations](#diffusers-specific-optimizations). Initially, no models are loaded onto the GPU and everything resides on the CPU. It requires about **10.3 GB** to keep all components on the CPU. However, the peak memory used during inference drops to **12.4 GB**. Note that inference will be slightly slower due to the time required to move different modeling components between CPU to GPU and back.

```diff
pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16,
- ).to("cuda")
+ )

+ pipe.enable_model_cpu_offload()
```

- Let's enable VAE tiling as described in [diffusers-specific optimizations](#diffusers-specific-optimizations) to further reduce memory usage at inference to `7.9` GB.

```diff
pipe = ...
pipe.enable_model_cpu_offload()

+ pipe.vae.enable_tiling()
```

Instead of `pipe.enable_model_cpu_offload()`, one can use `pipe.enable_sequential_cpu_offload()` that brings down memory usage to under **5 GB** without quantization. With quantization, there are some errors with the combination of accelerate and torchao. Better support can be expected soon. Note that this comes with an enormous slowdown of 3-5x in inference time from our tests.

#### Diffusers-specific optimizations

For supported architectures, memory requirements could further be brought down using Diffusers-supported functionality:
- `pipe.enable_model_cpu_offload()`: Only keeps the active Diffusers-used models (text encoder, transformer/unet, vae) on device
- `pipe.enable_sequential_cpu_offload()`: Similar to above, but performs cpu offloading more aggressively by only keeping active torch modules on device
- `pipe.vae.enable_vae_tiling()`: Enables tiled encoding/decoding by breaking up latents into smaller tiles and performing respective operation on each tile
- `pipe.vae.enable_vae_slicing()`: Helps keep memory usage constant when generating more than one image/video at a time

### Autoquant and autotuning

Given these many options around quantization, which one do I choose for my model? Enter ["autoquant"](https://github.com/pytorch/ao/tree/main/torchao/quantization#autoquantization). It tries quickly and accurately quantize your model. By the end of the process, it creates a "quantization plan" which can be accessed through `AUTOQUANT_CACHE` and reused. 

So, we would essentially do after performing quantization with autoquant and benchmarking:

```python
from torchao.quantization.autoquant import AUTOQUANT_CACHE
import pickle 

with open("quantization-cache.pkl", "wb") as f:
    pickle.dump(AUTOQUANT_CACHE)
```

And then to reuse the plan, we would do in our final codebase:

```python
from torchao.quantization.autoquant import AUTOQUANT_CACHE
with open("quantization-cache.pkl", "rb") as f:
    AUTOQUANT_CACHE.update(pickle.load(f))
```

Know more about "autoquant" [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#autoquantization). 

Another useful (but time-consuming) feature of `torchao` is ["autotuning"](https://github.com/pytorch/ao/tree/main/torchao/kernel). It tunes the `int_scaled_matmul` kernel for int8 dynamic + int8 weight quantization for the shape at runtime (given the shape of tensor passed to `int_scaled_matmul` op). Through this process, it tries to identify the most efficient kernel configurations for a given model and inputs.

To launch quantization benchmarking with autotuning, we need to enable the `TORCHAO_AUTOTUNER_ENABLE`. So, essentially: `TORCHAO_AUTOTUNER_ENABLE=1 TORCHAO_AUTOTUNER_DATA_PATH=my_data.pkl python my_script.py`. And when it's done, we can simply reuse the configs it found by doing: `TORCHAO_AUTOTUNER_DATA_PATH=my_data.pkl python my_script.py`. 

If you're using autotuning, keep in mind that it only works for intX quantization, for now and it is quite time-consuming. 

## Training with FP8

Check out the [`training`](./training/) directory.

## Serialization and loading quantized models

Check out our serialization and loading guide [here](./inference/serialization_and_loading.md). 

## Things to keep in mind when benchmarking

In this section, we provide a non-exhaustive overview of the things we learned during the benchmarking process. 

* *Expected gains and their ceiling are dependent on the hardware being used*. For example, compute density of the operations popped on a GPU has an effect on on the speedup. For the same code, you may see better numbers on an A100 than H100, simply because the operations weren't compute-dense enough for H100. In these situations, bigger batch sizes might make the effect of using a better GPU like H100 more pronounced.

* *Shapes matter*. Not all models are created equal. Certain shapes are friendlier in order for quantization to show its benefits over others. Usually, bigger shapes benefit quantization, resulting into speedups. The thinner the dimensions, the less pronounced the effects of quantization, especially for precisions like int8. In our case, using quantization on smaller models like [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) wasn't particularly beneficial. This is why, `torchao` provides an "autoquant" option that filters out smaller layers to exclude from quantization. 

* *Small matmuls.* If the matmuls of the underlying are small enough or the performance without quantization isn't bottlenecked by weight load time, these techniques may reduce performance.

* *Cache compilation results.* `torch.compile()` can take long just like any other deep-learning compiler. So, it is always recommended to cache the compilation results. Refer to [the official guide](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html) to know more. 

## Benefitting from `torch.compile()`

In this section, we provide a rundown of the scenarios that may prevent your model to optimally benefit from `torch.compile()`. This is very specific to `torch.compile()` and the `FluxPipeline.`

* Ensure there are no graph-breaks when `torch.compile()` is applied on the model. Briefly, graph-breaks introduce
unnecessary overheads blocking `torch.compile()` to obtain a full and dense graph of your model. In the case of Flux, we identified that it came from position embeddings, which was fixed in the following PRs: [#9307](https://github.com/huggingface/diffusers/pull/9307) and [#9321](https://github.com/huggingface/diffusers/pull/9321). Thanks to [Yiyi](https://github.com/yiyixuxu).

* Use the `torch.profiler.profile()` to get a kernel trace to identify if there is any graph break. You could use a script like [this](https://github.com/huggingface/diffusion-fast/blob/main/run_profile.py). This will give you a JSON file which you can upload to https://ui.perfetto.dev/ to view the trace. Additionally, use [this guide](https://pytorch.org/docs/stable/torch_cuda_memory.html) to validate the memory wins when using `torchao` for quantization and combining it with `torch.compile()`. 

* Finally, [this `torch.compile()` manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) is a gem of a reading to get an idea of how to go about approaching the profiling process.

## Acknowledgement

Thanks to the PyTorch AO team for help and guidance.