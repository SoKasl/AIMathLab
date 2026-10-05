#!/bin/bash
set -e

# Run Large-Scale 4,000-Problem Benchmark with KV Caching
VENV_PYTHON="./venv/bin/python"
EVAL_FILE="math_eval_4000.txt"

echo "================================================================="
echo "Starting Large-Scale 4,000-Problem Benchmark Evaluation"
echo "Evaluator Dataset: ${EVAL_FILE}"
echo "KV Cache: ENABLED (Accelerated Inference)"
echo "================================================================="

# 1. Run 9: Modern Dense (Hierarchical Curriculum)
echo ""
echo ">>> Evaluating Model 1/2: Run 9 - Modern Dense (Hierarchical Curriculum)..."
$VENV_PYTHON evaluate.py \
    --model_path run9_modern_hierarchical_final.pt \
    --eval_file ${EVAL_FILE} \
    --flash_attention \
    --swiglu \
    --rmsnorm \
    --rope \
    --use_kv_cache \
    --sample_log_path eval_4000_run9_dense_samples.log

# 2. Run 10: Sparse MoE (Hierarchical Curriculum)
echo ""
echo ">>> Evaluating Model 2/2: Run 10 - Sparse MoE Top-2 (Hierarchical Curriculum)..."
$VENV_PYTHON evaluate.py \
    --model_path run10_moe_hierarchical_final.pt \
    --eval_file ${EVAL_FILE} \
    --flash_attention \
    --swiglu \
    --rmsnorm \
    --rope \
    --moe \
    --use_kv_cache \
    --sample_log_path eval_4000_run10_moe_samples.log

echo ""
echo "================================================================="
echo "Large-Scale 4,000-Problem Benchmark Evaluation Complete!"
echo "Sample logs saved to:"
echo "  - eval_4000_run9_dense_samples.log"
echo "  - eval_4000_run10_moe_samples.log"
echo "================================================================="
