#!/bin/bash
set -e

VENV_PYTHON="./venv/bin/python"
EVAL_FILE="math_eval_4000.txt"

echo "================================================================="
echo "Evaluating Step 30,000 Checkpoints on 4,000-Problem Benchmark"
echo "================================================================="

# 1. Run 9: Modern Dense (Step 30,000 Best Model)
echo ""
echo ">>> Evaluating Model 1/2: Run 9 - Modern Dense (Step 30,000)..."
$VENV_PYTHON evaluate.py \
    --model_path run9_modern_hierarchical_best_model_wiki20000.pt \
    --eval_file ${EVAL_FILE} \
    --flash_attention \
    --swiglu \
    --rmsnorm \
    --rope \
    --use_kv_cache \
    --sample_log_path eval_4000_run9_dense_step30k.log

# 2. Run 10: Sparse MoE (Step 30,000 Best Model)
echo ""
echo ">>> Evaluating Model 2/2: Run 10 - Sparse MoE Top-2 (Step 30,000)..."
$VENV_PYTHON evaluate.py \
    --model_path run10_moe_hierarchical_best_model_wiki20000.pt \
    --eval_file ${EVAL_FILE} \
    --flash_attention \
    --swiglu \
    --rmsnorm \
    --rope \
    --moe \
    --use_kv_cache \
    --sample_log_path eval_4000_run10_moe_step30k.log

echo ""
echo "================================================================="
echo "Step 30,000 Benchmark Complete!"
echo "================================================================="
