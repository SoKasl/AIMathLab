import matplotlib.pyplot as plt
import numpy as np
import os

# Set modern scientific publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

palette = {
    'primary': '#1f77b4',     # Classic deep blue
    'accent': '#d62728',      # Vibrant crimson / red
    'secondary': '#2ca02c',   # Emerald green
    'purple': '#9467bd',      # Muted purple
    'gray': '#7f7f7f',        # Cool gray
    'dark': '#222222',
}

# ----------------------------------------------------
# Panel A: Length Generalization Collapse (4,000 Benchmark)
# ----------------------------------------------------
categories = ['Addition\n(In-Dist, <=3d)', 'Addition\n(OOD, 4-digit)', 
              'Subtraction\n(In-Dist, <=3d)', 'Subtraction\n(OOD, 4-digit)']
dense_vals = [99.22, 0.00, 99.42, 1.66]
moe_vals   = [98.83, 0.00, 99.42, 1.66]

x = np.arange(len(categories))
width = 0.35

rects1 = ax1.bar(x - width/2, dense_vals, width, label='Modern Dense (Step 50k)', color='#2b5c8f', edgecolor='black', linewidth=0.8, alpha=0.9)
rects2 = ax1.bar(x + width/2, moe_vals, width, label='Sparse MoE (Step 50k)', color='#e05a47', edgecolor='black', linewidth=0.8, alpha=0.9)

ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=palette['dark'])
ax1.set_title('(a) Length Generalization Collapse (4-Digit OOD)', fontsize=13, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=10.5)
ax1.set_ylim(0, 115)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

# Add value labels on top of bars
def autolabel(rects, ax, offset=1.5):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# Annotate collapse boundary
ax1.axvline(1.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.text(1.5, 105, 'Length Boundary\n(Unseen 4 Digits)', ha='center', va='center', 
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff2f2', edgecolor='#d62728', alpha=0.9), fontsize=9, fontweight='bold')

# ----------------------------------------------------
# Panel B: Catastrophic Developmental Drift (Long Division)
# ----------------------------------------------------
models = ['Modern Dense (Run 9)', 'Sparse MoE (Run 10)']
step30k = [86.67, 84.33]
step50k = [0.00, 1.67]

x2 = np.arange(len(models))
width2 = 0.32

rects_peak = ax2.bar(x2 - width2/2, step30k, width2, label='Step 30,000 (Peak Multi-Task)', color='#2e7d32', edgecolor='black', linewidth=0.8, alpha=0.9)
rects_final = ax2.bar(x2 + width2/2, step50k, width2, label='Step 50,000 (Overwritten / Drift)', color='#c62828', edgecolor='black', linewidth=0.8, alpha=0.9)

ax2.set_ylabel('Single-Digit Division Accuracy (%)', fontsize=12, fontweight='bold', color=palette['dark'])
ax2.set_title('(b) Catastrophic Drift on Single-Digit Division', fontsize=13, fontweight='bold', pad=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(models, fontsize=11, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

autolabel(rects_peak, ax2)
autolabel(rects_final, ax2)

# Annotate collapse
ax2.annotate('Catastrophic Drift:\n-86.7% Collapse', xy=(0 + width2/2, 2), xytext=(0.2, 45),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8e1', edgecolor='#f57f17', alpha=0.95),
             fontsize=9.5, fontweight='bold')

plt.tight_layout()

# Save both PDF and PNG to figures/
os.makedirs('paper_latex/figures', exist_ok=True)
pdf_path = 'paper_latex/figures/figure5_benchmark_dynamics.pdf'
png_path = 'paper_latex/figures/figure5_benchmark_dynamics.png'

plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight', dpi=300)
print(f"Successfully generated:\n  - {pdf_path}\n  - {png_path}")
