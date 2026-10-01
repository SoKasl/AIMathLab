#!/bin/bash

# Activate virtual environment with torch and dependencies
source venv/bin/activate

echo "Starting Hierarchical Developmental Curriculum Experiments..."

echo "================================================"
echo "RUN 9: Modern Architecture (Hierarchical Curriculum Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --use_hierarchical_data \
  --wiki_intervals 20000 \
  --math_eval_points 0 1000 2000 3000 4000 5000 7500 10000 15000 20000 30000 50000 \
  --csv_output "run9_modern_hierarchical.csv"

echo "================================================"
echo "RUN 10: Mixture of Experts Architecture (Hierarchical Curriculum Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --moe --use_hierarchical_data \
  --wiki_intervals 20000 \
  --math_eval_points 0 1000 2000 3000 4000 5000 7500 10000 15000 20000 30000 50000 \
  --csv_output "run10_moe_hierarchical.csv"

echo "================================================"
echo "HIERARCHICAL EXPERIMENTS COMPLETE!"
echo "================================================"
