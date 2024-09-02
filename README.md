# diffusers-torchao

Optimize image and video generation with [`diffusers`](https://github.com/huggingface/diffusers), [`torchao`](https://github.com/pytorch/ao), combining `torch.compile()` 🔥 We provide end-to-end inference and experimental training recipes to use `torchao` with `diffusers` in this repo. We demonstrate XX% speedup on [Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and YY% speedup on [Cog](https://huggingface.co/THUDM/CogVideoX-5b).

No-frills code:

```diff
from diffusers import FluxPipeline
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

This, alone, is sufficient to cut down inference time from X seconds to Y seconds on an H100. Check out the `inference` directory for the code.

<h4>Table of contents</h4>

* [Environment](#environment)
* [Benchmarking results](#benchmarking-results)
* [Training with FP8](#training-with-fp8)
* [Serialization and loading quantized models](#serialization-and-loading-quantized-models)
* [Things to keep in mind when benchmarking](#things-to-keep-in-mind-when-benchmarking)
* [Benefitting from `torch.compile()`](#benefitting-from-torchcompile)

## Environment

## Benchmarking results

We benchmark two models ([Flux.1-Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)) using different supported quantization datatypes in `torchao`. The results are as follows:

TODO: Find out what the best way of presenting all the information is. Having multiple giant table might be difficult to parse visually.

TODO: Make a note about ["autoquant"](https://github.com/pytorch/ao/tree/main/torchao/quantization#autoquantization) and "autotuning". 

## Training with FP8

Check out the `training` directory.

## Serialization and loading quantized models

Check out our serialization and loading guide [here](./inference/serialization_and_loading.md). 

## Things to keep in mind when benchmarking

In this section, we provide a non-exhaustive overview of the things we learned during the benchmarking process. 

* *Expected gains and their ceiling are dependent on the hardware being used*. For example, compute density of the operations popped on a GPU has an effect on on the speedup. For the same code, you may see better numbers on an A100 than H100, simply because the operations weren't compute-dense enough for H100. In these situations, bigger batch sizes might make the effect of using a better GPU like H100 more pronounced.

* *Shapes matter*. Not all models are created equal. Certain shapes are friendlier in order for quantization to show its benefits over others. Usually, bigger shapes benefit quantization, resulting into speedups. The thinner the dimensions, the less pronounced the effects of quantization, especially for precisions like int8. In our case, using quantization on smaller models like [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) wasn't particularly beneficial. 

* *Small matmuls.* If the matmuls of the underlying are small enough or the performance without quantization isn't bottlenecked by weight load time, these techniques may reduce performance.

## Benefitting from `torch.compile()`

In this section, we provide a rundown of the scenarios that may prevent your model to optimally benefit from `torch.compile()`. This is very specific to `torch.compile()` and the `FluxPipeline.`

* Ensure there are no graph-breaks when `torch.compile()` is applied on the model. Briefly, graph-breaks introduce
unnecessary overheads blocking `torch.compile()` to obtain a full and dense graph of your model. In the case of Flux, we identified that it came from position embeddings, which was fixed in the following PRs: [#9307](https://github.com/huggingface/diffusers/pull/9307) and [#9321](https://github.com/huggingface/diffusers/pull/9321). Thanks to [Yiyi](https://github.com/yiyixuxu).

* Use the `torch.profiler.profile()` to get a kernel trace to identify if there is any graph break. You could use a script like [this](https://github.com/huggingface/diffusion-fast/blob/main/run_profile.py). This will give you a JSON file which you can upload to https://ui.perfetto.dev/ to view the trace. 

* Finally, [this `torch.compile()` manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) is a gem of a reading to get an idea of how to go about approaching the profiling process.