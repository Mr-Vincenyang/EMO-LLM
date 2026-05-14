#!/usr/bin/env bash
# Train all 4 style LoRAs in parallel across 2 GPUs.
#
# GPU 0: empathetic + rational
# GPU 1: encouraging + calm_safe
#
# Each training uses ~16GB VRAM, so 2 per GPU fits on A100 80GB.
#
# Usage: bash scripts/train_parallel.sh

set -euo pipefail

LOG_DIR="log"
mkdir -p "$LOG_DIR" outputs/lora

echo "========================================="
echo "Launching 4 parallel LoRA trainings"
echo "GPU 0: empathetic + rational"
echo "GPU 1: encouraging + calm_safe"
echo "========================================="
echo ""

START_TIME=$(date +%s)

# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/train_style.py --style empathetic &
PID1=$!
CUDA_VISIBLE_DEVICES=0 python scripts/train_style.py --style rational &
PID2=$!

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/train_style.py --style encouraging &
PID3=$!
CUDA_VISIBLE_DEVICES=1 python scripts/train_style.py --style calm_safe &
PID4=$!

echo "Processes: empathetic=$PID1 rational=$PID2 encouraging=$PID3 calm_safe=$PID4"
echo ""

# Wait for all
wait $PID1; echo "empathetic done (exit $?)"
wait $PID2; echo "rational done (exit $?)"
wait $PID3; echo "encouraging done (exit $?)"
wait $PID4; echo "calm_safe done (exit $?)"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================="
echo "All 4 LoRAs trained in ${ELAPSED}s"
echo "========================================="

# Collect metrics
echo ""
echo "--- Per-style metrics ---"
for style in empathetic rational encouraging calm_safe; do
    metrics_file=$(ls -t log/metrics_${style}_*.json 2>/dev/null | head -1)
    if [ -n "$metrics_file" ]; then
        python3 -c "
import json
with open('$metrics_file') as f:
    m = json.load(f)
print(f\"  {m['style']:15s} | loss={m['train_loss']:.4f} | time={m['train_time_s']:.0f}s | peak_gpu={m['gpu_mem_peak_gb']:.1f}GB | samples={m['samples']}\")
"
    fi
done
