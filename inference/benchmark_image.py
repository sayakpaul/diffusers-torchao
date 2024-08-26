import torch

torch.set_float32_matmul_precision("high")

import torch.utils.benchmark as benchmark
from diffusers import DiffusionPipeline
import gc

from torchao.quantization import (
    int4_weight_only,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    quantize_,
    autoquant,
)
from torchao.float8.inference import ActivationCasting, QuantConfig, quantize_to_float8
from torchao.prototype.quant_llm import fp6_llm_weight_only
from torchao.sparsity import sparsify_, int8_dynamic_activation_int8_semi_sparse_weight
from tabulate import tabulate
import argparse
import json


PROMPT = "Eiffel Tower was Made up of more than 2 million translucent straws to look like a cloud, with the bell tower at the top of the building, Michel installed huge foam-making machines in the forest to blow huge amounts of unpredictable wet clouds in the building's classic architecture."
PREFIXES = {
    "stabilityai/stable-diffusion-3-medium-diffusers": "sd3",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "pixart",
    "fal/AuraFlow": "auraflow",
    "black-forest-labs/FLUX.1-dev": "flux-dev",
}


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


def load_pipeline(
    ckpt_id: str,
    fuse_attn_projections: bool,
    compile: bool,
    quantization: str,
    sparsify: bool,
    compile_vae: bool = False,
) -> DiffusionPipeline:
    pipeline = DiffusionPipeline.from_pretrained(ckpt_id, torch_dtype=torch.bfloat16).to("cuda")

    if fuse_attn_projections:
        pipeline.transformer.fuse_qkv_projections()
        if compile_vae:
            pipeline.vae.fuse_qkv_projections()

    if quantization == "autoquant" and compile:
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        if compile_vae:
            pipeline.vae.to(memory_format=torch.channels_last)
            pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

    if not sparsify:
        if quantization == "int8dq":
            quantize_(pipeline.transformer, int8_dynamic_activation_int8_weight())
            if compile_vae:
                quantize_(pipeline.vae, int8_dynamic_activation_int8_weight())
        elif quantization == "int8wo":
            quantize_(pipeline.transformer, int8_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, int8_weight_only())
        elif quantization == "int4wo":
            quantize_(pipeline.transformer, int4_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, int4_weight_only())
        elif quantization == "fp6":
            quantize_(pipeline.transformer, fp6_llm_weight_only())
            if compile_vae:
                quantize_(pipeline.vae, fp6_llm_weight_only())
        elif quantization == "fp8":
            pipeline.transformer = quantize_to_float8(pipeline.transformer, QuantConfig(ActivationCasting.DYNAMIC))
            if compile_vae:
                pipeline.vae = quantize_to_float8(pipeline.vae, QuantConfig(ActivationCasting.DYNAMIC))
        elif quantization == "autoquant":
            pipeline.transformer = autoquant(pipeline.transformer, error_on_unseen=False)
            if compile_vae:
                pipeline.vae = autoquant(pipeline.vae, error_on_unseen=False)

    if sparsify:
        sparsify_(pipeline.transformer, int8_dynamic_activation_int8_semi_sparse_weight())
        if compile_vae:
            sparsify_(pipeline.vae, int8_dynamic_activation_int8_semi_sparse_weight())

    if quantization != "autoquant" and compile:
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
        if compile_vae:
            pipeline.vae.to(memory_format=torch.channels_last)
            pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run_inference(pipe, batch_size):
    _ = pipe(
        prompt=PROMPT,
        num_images_per_prompt=batch_size,
        generator=torch.manual_seed(2024),
    )


def pretty_print_results(results, precision: int = 6):
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return value

    filtered_table = {k: format_value(v) for k, v in results.items()}
    print(tabulate([filtered_table], headers="keys", tablefmt="pipe", stralign="center"))


def run_benchmark(pipeline, args):
    for _ in range(5):
        run_inference(pipeline, batch_size=args.batch_size)

    time = benchmark_fn(run_inference, pipeline, args.batch_size)
    torch.cuda.empty_cache()
    memory = bytes_to_giga_bytes(torch.cuda.memory_allocated())  # in GBs.

    info = dict(
        ckpt_id=args.ckpt_id,
        batch_size=args.batch_size,
        fuse=args.fuse_attn_projections,
        compile=args.compile,
        compile_vae=args.compile_vae,
        quantization=args.quantization,
        sparsify=args.sparsify,
        memory=memory,
        time=time,
    )

    pretty_print_results(info)
    return info


def serialize_artifacts(info: dict, pipeline, args):
    ckpt_id = PREFIXES[args.ckpt_id]
    prefix = f"ckpt@{ckpt_id}-bs@{args.batch_size}-fuse@{args.fuse_attn_projections}-compile@{args.compile}-compile_vae@{args.compile_vae}-quant@{args.quantization}-sparsify@{args.sparsify}"
    info_file = f"{prefix}_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f)

    image = pipeline(
        prompt=PROMPT,
        num_images_per_prompt=args.batch_size,
        generator=torch.manual_seed(0),
    ).images[0]
    image.save(f"{prefix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_id",
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        type=str,
        help="Hub model or path to local model for which the benchmark is to be run."
    )
    parser.add_argument(
        "--fuse_attn_projections",
        action="store_true",
        help="Whether or not to fuse the QKV projection layers into one larger layer.",
    )
    parser.add_argument("--compile", action="store_true", help="Whether or not to torch.compile the models.")
    parser.add_argument("--compile_vae", action="store_true", help="If compiling, should VAE be compiled too?")
    parser.add_argument(
        "--quantization",
        default="None",
        choices=["int8dq", "int8wo", "int4wo", "autoquant", "fp6", "fp8", "None"],
        help="Which quantization technique to apply",
    )
    parser.add_argument("--sparsify", action="store_true")
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        choices=[1, 4, 8],
        help="Number of images to generate for the testing prompt."
    )
    args = parser.parse_args()

    flush()

    pipeline = load_pipeline(
        ckpt_id=args.ckpt_id,
        fuse_attn_projections=args.fuse_attn_projections,
        compile=args.compile,
        compile_vae=args.compile_vae,
        quantization=args.quantization,
        sparsify=args.sparsify,
    )

    info = run_benchmark(pipeline, args)
    serialize_artifacts(info, pipeline, args)
