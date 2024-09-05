# Training DreamBooth LoRAs on Flux.1-Dev with FP8

> [!IMPORTANT]  
> Since we are utilizing FP8 tensor cores we need CUDA GPUs with compute capability at least 8.9 or greater. If you're looking for memory-efficient training on relatively older cards, we encourage you to check out other trainers like [SimpleTuner](https://github.com/bghira/SimpleTuner), [ai-toolkit](https://github.com/ostris/ai-toolkit/), etc.

Please refer to [this document](https://gist.github.com/sayakpaul/f0358dd4f4bcedf14211eba5704df25a) for full code. It shows end-to-end training, serialization, and inference. It is basically modification of the original `diffusers` [DreamBooth LoRA training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py) (for Flux.1-Dev) for enabling FP8 training support. 

## Summary of the changes required

**First inject FP8 layers in the model you're training**

```diff
+ from torchao.float8 import convert_to_float8_training, Float8LinearConfig

+ convert_to_float8_training(
+    module_being_trained, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
+)
```

What is `module_filter_fn`?

It is function that helps to filter out the modules that should not be injected with FP8 layers. 

```python
def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True
```

**Then add LoRA config (if applicable)**

```python
lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.rank,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
module_being_trained.add_adapter(lora_config)
```

Make sure to convert the LoRA layers to FP32 for stability.

## Reference

For all the knobs: https://github.com/pytorch/ao/tree/main/torchao/float8.

