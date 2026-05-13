#!/usr/bin/env bash
# Interactive controllable generation using trained LoRA adapters.
# Usage: bash scripts/generate.sh

set -euo pipefail

MODEL="Qwen/Qwen3-1.5B"
LORA_DIR="outputs/lora"

echo "Starting controllable generation with style interpolation..."
echo "Model: $MODEL"
echo ""

python src/generate.py \
    --model_name "$MODEL" \
    --lora_paths "empathetic=${LORA_DIR}/empathetic,rational=${LORA_DIR}/rational,encouraging=${LORA_DIR}/encouraging,calm_safe=${LORA_DIR}/calm_safe" \
    --mode weight \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_new_tokens 512
