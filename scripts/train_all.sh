#!/usr/bin/env bash
# Train all 4 style LoRAs sequentially.
# Usage: bash scripts/train_all.sh [BASE_MODEL] [DATA_DIR]

set -euo pipefail

BASE_MODEL="${1:-Qwen/Qwen3-1.5B}"
DATA_DIR="${2:-data/train}"

STYLES=("empathetic" "rational" "encouraging" "calm_safe")

for style in "${STYLES[@]}"; do
    echo "========================================="
    echo "Training LoRA for style: $style"
    echo "========================================="

    python src/train_lora.py \
        --style "$style" \
        --model_name "$BASE_MODEL" \
        --data_path "$DATA_DIR/${style}.jsonl" \
        --output_dir "outputs/lora/${style}" \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 3 \
        --learning_rate 2.0e-4 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --save_strategy epoch \
        --fp16 True \
        --seed 42

    echo "Finished training: $style"
    echo ""
done

echo "All 4 LoRA adapters trained successfully."
echo "Outputs saved to: outputs/lora/"
ls -la outputs/lora/
