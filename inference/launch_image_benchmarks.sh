#!/bin/bash

# Possible values for each argument
ckpt_ids=("black-forest-labs/FLUX.1-dev")
fuse_attn_projections_flags=("" "--fuse_attn_projections")
compile_flags=("" "--compile")
quantizations=("int8dq" "int8wo" "fp8wo" "fp8dq" "None")
sparsify_flags=("")
batch_sizes=(1 4 8)

# Loop over all combinations
for ckpt_id in "${ckpt_ids[@]}"; do
  for quantization in "${quantizations[@]}"; do
    # Determine the valid flags based on quantization value
    if [ "$quantization" == "None" ]; then
      sparsify_flags=("" "--sparsify")  # Enable sparsify
      fuse_attn_projections_flags=("")  # Disable fuse_attn_projections
    else
      sparsify_flags=("")  # Disable sparsify
      fuse_attn_projections_flags=("" "--fuse_attn_projections")  # Enable fuse_attn_projections
    fi
    
    for fuse_attn_projections in "${fuse_attn_projections_flags[@]}"; do
      for compile in "${compile_flags[@]}"; do
        for sparsify in "${sparsify_flags[@]}"; do
          for batch_size in "${batch_sizes[@]}"; do
            # Construct the command
            cmd="python3 benchmark_image.py --ckpt_id \"$ckpt_id\" $fuse_attn_projections $compile --quantization \"$quantization\" $sparsify --batch_size \"$batch_size\""
            
            # Echo the command
            echo "Running command: $cmd"
            
            # Run the command
            eval $cmd
          done
        done
      done
    done
  done
done
