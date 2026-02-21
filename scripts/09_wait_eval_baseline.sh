#!/bin/bash
# Wait for baseline training to complete, then run evaluation
set -e

CKPT_DIR="/home/mekashirskiy/rom4ik/doom_diff/checkpoints/baseline"
FINAL_CKPT="${CKPT_DIR}/memgamengen_final.pt"
PROJECT_DIR="/home/mekashirskiy/rom4ik/doom_diff"

echo "Waiting for baseline training to produce final checkpoint..."
echo "Checking for: ${FINAL_CKPT}"

while [ ! -f "${FINAL_CKPT}" ]; do
    echo "$(date): Waiting... Current checkpoints:"
    ls ${CKPT_DIR}/*.pt 2>/dev/null || echo "  none yet"
    sleep 120
done

echo "$(date): Final checkpoint found! Running evaluation..."

cd "${PROJECT_DIR}"
CUDA_VISIBLE_DEVICES=7 python scripts/07_evaluate_baseline.py 2>&1

echo "$(date): Baseline evaluation complete!"
echo "Results saved to: results/baseline_evaluation_results.json"
