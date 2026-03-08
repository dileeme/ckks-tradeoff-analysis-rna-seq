"""
Plot 3 — Ciphertext size vs polynomial modulus degree, BFV vs CKKS.

CT size is batch-independent: it depends only on poly modulus degree (N)
and the coefficient modulus chain, not on the number of samples encrypted.
Scaling per PMD doubling:
  BFV  ×4.00 — coeff modulus grows proportionally with N (integer arithmetic)
  CKKS ×3.15 — coeff modulus chain structure is sparser (fewer rescue primes)

Storage implication for encrypted genomics pipelines:
  At N=16384: BFV ~835 KB/sample, CKKS ~264 KB/sample.
  For 800 samples: BFV ≈ 668 MB total, CKKS ≈ 211 MB total.
  Both are ~10–40× plaintext footprint; must be accounted for in cloud
  storage budgets and network transfer costs.

Results averaged across 10 independent runs (std negligible for CT size).

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

ct = (all_data
    .groupby(['scheme', 'poly_mod_degree'])['ct_size_kb']
    .agg(mean='mean', std='std').reset_index())

bfv_ct  = ct[ct['scheme'] == 'BFV'].sort_values('poly_mod_degree')
ckks_ct = ct[ct['scheme'] == 'CKKS'].sort_values('poly_mod_degree')

bfv_ratios  = bfv_ct['mean'].values[1:] / bfv_ct['mean'].values[:-1]
ckks_ratios = ckks_ct['mean'].values[1:] / ckks_ct['mean'].values[:-1]

fig, ax = plt.subplots(figsize=(7, 5))

for scheme, df_ct in [('BFV', bfv_ct), ('CKKS', ckks_ct)]:
    pmds = df_ct['poly_mod_degree'].values
    key  = (scheme, pmds[-1])
    ax.plot(pmds, df_ct['mean'],
            color=COLORS[key], marker=MARKERS[key],
            linestyle=LINESTYLES[scheme], linewidth=LW, markersize=8,
            label=scheme, zorder=3)
    ax.fill_between(pmds,
                    df_ct['mean'] - df_ct['std'],
                    df_ct['mean'] + df_ct['std'],
                    color=COLORS[key], alpha=BAND_ALPHA)

# ── Ratio annotations at geometric midpoints ─────────────────────────────
bfv_pmds = bfv_ct['poly_mod_degree'].values
bfv_vals = bfv_ct['mean'].values
for i, ratio in enumerate(bfv_ratios):
    mid_x = np.sqrt(bfv_pmds[i] * bfv_pmds[i + 1])
    mid_y = np.sqrt(bfv_vals[i] * bfv_vals[i + 1]) * 1.18
    ax.text(mid_x, mid_y, f'×{ratio:.2f}', ha='center', va='bottom',
            fontsize=8.5, color=COLORS[('BFV', 16384)], fontweight='bold')

ckks_pmds = ckks_ct['poly_mod_degree'].values
ckks_vals = ckks_ct['mean'].values
for i, ratio in enumerate(ckks_ratios):
    mid_x = np.sqrt(ckks_pmds[i] * ckks_pmds[i + 1])
    mid_y = np.sqrt(ckks_vals[i] * ckks_vals[i + 1]) * 1.18
    ax.text(mid_x, mid_y, f'×{ratio:.2f}', ha='center', va='bottom',
            fontsize=8.5, color=COLORS[('CKKS', 16384)], fontweight='bold')

# ── KB value labels ───────────────────────────────────────────────────────
for _, row in bfv_ct.iterrows():
    ax.annotate(f"{row['mean']:.0f} KB",
                xy=(row['poly_mod_degree'], row['mean']),
                xytext=(0, -16), textcoords='offset points',
                ha='center', fontsize=8, color=COLORS[('BFV', 16384)])
for _, row in ckks_ct.iterrows():
    ax.annotate(f"{row['mean']:.0f} KB",
                xy=(row['poly_mod_degree'], row['mean']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, color=COLORS[('CKKS', 16384)])

# ── Axes: both log₂ ──────────────────────────────────────────────────────
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.set_xticks([4096, 8192, 16384])
ax.set_xticklabels(['4096\n(BFV only)', '8192', '16384'])
ax.set_yticks([128, 256, 512, 1024, 2048])
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

ax.set_xlabel('Polynomial modulus degree (N)', fontsize=10)
ax.set_ylabel('Ciphertext size (KB, log₂ scale)', fontsize=10)
ax.set_title(
    'Ciphertext Size vs Polynomial Modulus Degree\n'
    'BFV vs CKKS — Averaged Across All Samples and Runs\n'
    'CT size is batch-independent; scaling per PMD doubling annotated',
    fontsize=10.5, fontweight='bold', pad=10)
style_ax(ax)
ax.legend(fontsize=9.5, title='Scheme', title_fontsize=9,
          loc='upper left', **LEGEND_KW)

# Caption text (copy into LaTeX):
#   "Ciphertext (CT) size in KB on a log₂–log₂ axis. CT size depends only
#    on poly modulus degree N and is independent of sample count (batch size).
#    BFV scales ×4.00 per PMD doubling; CKKS scales ×3.15, reflecting the
#    sparser coefficient modulus chain at each security level. Value labels
#    show mean KB. Std is negligible (<1 KB) and not shown."

add_env_note(fig)
plt.tight_layout()
plt.savefig('figures/plot3_ct_size.png', dpi=180, bbox_inches='tight')
plt.show()
print("Saved figures/plot3_ct_size.png")
