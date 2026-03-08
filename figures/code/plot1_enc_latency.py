"""
Plot 1 — Encryption latency vs sample count, BFV vs CKKS.

Enc latency = time to encode and encrypt all samples in the batch.
Each sample is encrypted as one independent ciphertext, so latency scales
approximately linearly with sample count for both schemes.

Results averaged across 10 independent runs; ±1 std shown as shaded bands.

Hardware: Intel i7-10750H, 16 GB RAM, Ubuntu 22.04 (WSL2)
Software: Python 3.12, TenSEAL 0.3.14

HE parameters:
  CKKS — N=8192:  coeff_mod=[40,30,30,40], scale=2^30
  CKKS — N=16384: coeff_mod=[60,40,40,60], scale=2^40
  BFV  — N=4096/8192/16384: default SEAL integer params
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plot_style import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

bfv1  = pd.read_csv('results/phase3_bfv_dataset1.csv')
bfv2  = pd.read_csv('results/phase3_bfv_dataset2.csv')
ckks1 = pd.read_csv('results/phase3_ckks_dataset1.csv')
ckks2 = pd.read_csv('results/phase3_ckks_dataset2.csv')
all_data = pd.concat([bfv1, bfv2, ckks1, ckks2], ignore_index=True)

stats = (all_data
    .groupby(['dataset', 'scheme', 'poly_mod_degree', 'samples'])['enc_latency_ms']
    .agg(mean='mean', std='std').reset_index())

datasets = [
    ('dataset1', 'Dataset 1 — UCI RNA-Seq\n10 cancer pairs, n ∈ {100, 400, 801}'),
    ('dataset2', 'Dataset 2 — TCGA LUSC+LUAD\n1 cancer pair, n ∈ {100, 400, 1129}'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.32)

for ax, (ds, title) in zip(axes, datasets):
    sub = stats[stats['dataset'] == ds]

    for (scheme, pmd), grp in sub.groupby(['scheme', 'poly_mod_degree']):
        grp = grp.sort_values('samples')
        key = (scheme, pmd)
        ax.plot(grp['samples'], grp['mean'],
                color=COLORS[key], marker=MARKERS[key],
                linestyle=LINESTYLES[scheme], linewidth=LW, markersize=MS,
                label=scheme_label(scheme, pmd), zorder=3)
        ax.fill_between(grp['samples'],
                        grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'],
                        color=COLORS[key], alpha=BAND_ALPHA, zorder=2)

    # log₁₀ scale on y-axis
    ax.set_yscale('log', base=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=9)
    ax.set_xlabel('Sample count (n)', fontsize=10)
    ax.set_ylabel('Enc latency (ms, log₁₀ scale)', fontsize=9.5)
    style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (
        0 if 'CKKS' in labels[i] else 1,
        -int(labels[i].split('=')[1].rstrip(')'))
    ))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=8.2, loc='upper left',
              title='Scheme (poly mod degree N)', title_fontsize=8,
              **LEGEND_KW)

# Caption-level interpretation goes in the paper, not in the figure.
# Caption text (copy into LaTeX):
#   "Encryption latency (enc\_latency\_ms) = time to encode and encrypt all
#    samples; one independent ciphertext per sample. Latency scales
#    approximately linearly with sample count for both BFV and CKKS because
#    each encryption operation is independent. Shaded bands = ±1 std across
#    10 runs. BFV N=4096 is shown only in Dataset 1 (Dataset 2 batch sizes
#    exceed available slot capacity at N=4096)."

fig.suptitle('Encryption Latency vs Sample Count — BFV vs CKKS',
             fontsize=12, fontweight='bold', y=1.01)
add_env_note(fig)
plt.tight_layout()
plt.savefig('figures/plot1_enc_latency.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved figures/plot1_enc_latency.png")
