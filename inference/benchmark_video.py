import argparse
import json
import os

os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"

import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
from torchao.quantization import (
    autoquant,
    quantize_,
    int8_weight_only,
    int8_dynamic_activation_int8_weight,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int4_weight_only,
    float8_dynamic_activation_float8_weight,
    float8_weight_only
)
from torchao.sparsity import sparsify_
from torchao.prototype.quant_llm import fp6_llm_weight_only

from utils import benchmark_fn, pretty_print_results, print_memory, reset_memory

# Set high precision for float32 matrix multiplications.
# This setting optimizes performance on NVIDIA GPUs with Ampere architecture (e.g., A100, RTX 30 series) or newer.
torch.set_float32_matmul_precision("high")


CONVERT_DTYPE = {
    "fp16": lambda module: module.to(dtype=torch.float16),
    "bf16": lambda module: module.to(dtype=torch.bfloat16),
    "fp8wo": lambda module: quantize_(module, float8_weight_only()),
    "fp8dq": lambda module: quantize_(module, float8_dynamic_activation_float8_weight()),
    "fp6": lambda module: quantize_(module, fp6_llm_weight_only()),
    "int8wo": lambda module: quantize_(module, int8_weight_only()),
    "int8dq": lambda module: quantize_(module, int8_dynamic_activation_int8_weight()),
    "int4dq": lambda module: quantize_(module, int8_dynamic_activation_int4_weight()),
    "int4wo": lambda module: quantize_(module, int4_weight_only()),
    "autoquant": lambda module: autoquant(module, error_on_unseen=False),
    "sparsify": lambda module: sparsify_(module, int8_dynamic_activation_int8_semi_sparse_weight()),
}


def load_pipeline(model_id, dtype, device, quantize_vae, compile, fuse_qkv):
    # 1. Load pipeline
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    if fuse_qkv:
        pipe.fuse_qkv_projections()

    # 2. Quantize and compile
    if dtype == "autoquant" and compile:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        # VAE cannot be compiled due to: https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f#file-test_cogvideox_torch_compile-py-L30

    text_encoder_return = CONVERT_DTYPE[dtype](pipe.text_encoder)
    transformer_return = CONVERT_DTYPE[dtype](pipe.transformer)
    vae_return = None
    if dtype in ["fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"] or quantize_vae:
        vae_return = CONVERT_DTYPE[dtype](pipe.vae)

    if text_encoder_return is not None:
        pipe.text_encoder = text_encoder_return
    if transformer_return is not None:
        pipe.transformer = transformer_return
    if vae_return is not None:
        pipe.vae = vae_return

    if dtype != "autoquant" and compile:
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        # VAE cannot be compiled due to: https://gist.github.com/a-r-r-o-w/5183d75e452a368fd17448fcc810bd3f#file-test_cogvideox_torch_compile-py-L30

    return pipe


def run_inference(pipe):
    prompt = (
        "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        "atmosphere of this unique musical performance."
    )
    guidance_scale = 6
    num_inference_steps = 50

    video = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=True,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(3047),  # https://arxiv.org/abs/2109.08203
    )
    return video


def main(model_id, dtype, device, quantize_vae, compile, fuse_qkv):
    reset_memory(device)

    # 1. Load pipeline
    pipe = load_pipeline(model_id, dtype, device, quantize_vae, compile, fuse_qkv)

    print_memory(device)

    torch.cuda.empty_cache()
    model_memory = round(torch.cuda.memory_allocated() / 1024**3, 3)

    # 2. Warmup
    num_warmups = 2
    for _ in range(num_warmups):
        video = run_inference(pipe)

    # 3. Benchmark
    time = benchmark_fn(run_inference, pipe)
    print_memory(device)

    torch.cuda.empty_cache()
    inference_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    # 4. Save results
    model_type = "5B" if "5b" in model_id else "2B"
    info = {
        "model_type": model_type,
        "compile": compile,
        "fuse_qkv": fuse_qkv,
        "quantize_vae": quantize_vae,
        "quantization": dtype,
        "model_memory": model_memory,
        "inference_memory": inference_memory,
        "time": time,
    }
    pretty_print_results(info, precision=3)

    # Serialize artifacts
    model_name = model_id.replace("/", "_").replace(".", "_")
    filename_prefix = f"output-model_{model_name}-quantization_{dtype}-compile_{compile}-fuse_qkv_{fuse_qkv}"

    with open(f"{filename_prefix}.json", "w") as file:
        json.dump(info, file)

    export_to_video(video.frames[0], f"{filename_prefix}.mp4", fps=8)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="THUDM/CogVideoX-5b",
        choices=["THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"],
        help="Hub model or path to local model for which the benchmark is to be run.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=[
            "fp16",
            "bf16",
            "fp8wo",
            "fp8dq",
            "fp6",
            "int8wo",
            "int8dq",
            "int4dq",
            "int4wo",
            "autoquant",
            "sparsify",
        ],
        help="Inference or Quantization type.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument(
        "--quantize_vae",
        action="store_true",
        default=False,
        help="Whether or not to quantize the CogVideoX VAE. Can lead to worse decoding results in some quantization cases.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Whether or not to torch.compile the models. For our experiments with CogVideoX, we only compile the transformer.",
    )
    parser.add_argument(
        "--fuse_qkv",
        action="store_true",
        default=False,
        help="Whether or not to fuse the QKV projection layers into one larger layer.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    main(args.model_id, args.dtype, args.device, args.quantize_vae, args.compile, args.fuse_qkv)
