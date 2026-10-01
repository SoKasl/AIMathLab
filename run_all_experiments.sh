#!/bin/bash

# Activate virtual environment with torch and dependencies
source venv/bin/activate

echo "Starting FULL RERUN of Architectural Ablations on New CoT and Curriculum Data..."

echo "================================================"
echo "RUN 1: Baseline GPT-2 (Standard FOIL CoT Data)"
echo "================================================"
python3 experiment.py \
  --wiki_intervals 10 5000 10000 20000 \
  --math_eval_points 0 1000 2000 5000 \
  --csv_output "run1_baseline_new_cot.csv"

echo "================================================"
echo "RUN 2: Flash Attention (Standard FOIL CoT Data)"
echo "================================================"
python3 experiment.py --flash_attention \
  --wiki_intervals 10 5000 10000 20000 50000 \
  --math_eval_points 0 1000 2000 5000 \
  --csv_output "run2_flash_attention_new_cot.csv"

echo "================================================"
echo "RUN 3: Modern Architecture (Standard FOIL CoT Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope \
  --wiki_intervals 10 5000 10000 20000 50000 \
  --math_eval_points 0 1000 2000 5000 \
  --csv_output "run3_modern_new_cot.csv"

echo "================================================"
echo "RUN 4: Mixture of Experts (Standard FOIL CoT Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --moe \
  --wiki_intervals 10 10000 20000 50000 \
  --math_eval_points 0 1000 2000 5000 \
  --csv_output "run4_moe_new_cot.csv"

echo "================================================"
echo "RUN 5: Modern Architecture (Extended FOIL Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope \
  --wiki_intervals 20000 \
  --math_eval_points 10000 25000 50000 75000 100000 \
  --csv_output "run5_modern_extended_foil.csv"

echo "================================================"
echo "RUN 6: Mixture of Experts (Extended FOIL Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --moe \
  --wiki_intervals 20000 \
  --math_eval_points 10000 25000 50000 75000 100000 \
  --csv_output "run6_moe_extended_foil.csv"

echo "================================================"
echo "RUN 7: Modern Architecture (Curriculum Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --use_curriculum_data \
  --wiki_intervals 20000 \
  --math_eval_points 1000 2000 3000 4000 5000 10000 25000 50000 75000 100000 \
  --csv_output "run7_modern_curriculum.csv"

echo "================================================"
echo "RUN 8: Mixture of Experts Architecture (Curriculum Data)"
echo "================================================"
python3 experiment.py --flash_attention --swiglu --rmsnorm --rope --moe --use_curriculum_data \
  --wiki_intervals 20000 \
  --math_eval_points 1000 2000 3000 4000 5000 10000 25000 50000 75000 100000 \
  --csv_output "run8_moe_curriculum.csv"

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
echo "ALL EXPERIMENTS COMPLETE!"
echo "================================================"
