#!/bin/bash
set -e

python stage1_timing_eval_multi_pruner.py \
    --hf-dataset lmms-lab/VQAv2 \
    --hf-split validation \
    --num-samples 500 \
    --pruner-type llm-attention \
    --keep-ratio 0.5 \
    --llm-layers 1 \
    --streaming \
    --output results_llm_attention.json

python stage1_timing_eval_multi_pruner.py \
    --hf-dataset lmms-lab/VQAv2 \
    --hf-split validation \
    --num-samples 500 \
    --pruner-type clip-attention \
    --keep-ratio 0.5 \
    --streaming \
    --output results_clip_attention.json
