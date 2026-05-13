#!/usr/bin/env bash
# Launch Gradio demo for interactive style interpolation.
# Usage: bash scripts/run_demo.sh

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-1.5B}"
LORA_DIR="${2:-outputs/lora}"
PORT="${3:-7860}"

echo "Launching Gradio demo..."
echo "Model: $MODEL"
echo "LoRA dir: $LORA_DIR"
echo "Port: $PORT"
echo ""

python demo/app.py \
    --model_name "$MODEL" \
    --lora_dir "$LORA_DIR" \
    --port "$PORT"
