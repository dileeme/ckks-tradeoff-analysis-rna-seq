"""
Shared style constants for all BFV vs CKKS plots.
Import this in every plot script to guarantee consistency.

Hardware/software environment (reported in all figures):
  CPU  : Intel Core i7-10750H @ 2.60 GHz (6 cores / 12 threads)
  RAM  : 16 GB DDR4
  OS   : Ubuntu 22.04 LTS (WSL2 on Windows 11)
  Python: 3.12 | TenSEAL 0.3.14 | NumPy 1.26 | pandas 2.2

HE parameter table (used across all experiments):
  ┌────────────┬──────────────────────────┬────────────────┬───────────┐
  │  Scheme    │  Poly Mod Degree (N)     │  Coeff Mod     │  Scale    │
  ├────────────┼──────────────────────────┼────────────────┼───────────┤
  │  CKKS      │  8192                    │  [40,30,30,40] │  2^30     │
  │  CKKS      │  16384                   │  [60,40,40,60] │  2^40     │
  │  BFV       │  4096                    │  default       │  N/A      │
  │  BFV       │  8192                    │  default       │  N/A      │
  │  BFV       │  16384                   │  default       │  N/A      │
  └────────────┴──────────────────────────┴────────────────┴───────────┘

All metrics averaged across 10 independent runs; ±1 std shown as shaded bands.

Latency component definitions:
  enc_latency_ms  : Time to encode and encrypt all samples in the batch
                    (each sample → one independent ciphertext; scales ~linearly
                     with sample count for both schemes).
  exec_latency_ms : Time to perform all homomorphic operations — group-sum
                    accumulation + ciphertext subtraction for every pairwise
                    cancer-type comparison. Driven by number of pairs, NOT
                    sample count (CKKS uses SIMD batching; BFV operates
                    element-wise per ciphertext — see SIMD note below).
  dec_latency_ms  : Time to decrypt and decode all result ciphertexts.
                    Negligible relative to enc and exec for both schemes.

CKKS vs BFV batching behaviour:
  CKKS packs up to N/2 floating-point values into a single ciphertext via
  SIMD batching (N = poly modulus degree). One HE addition operates on the
  entire packed vector simultaneously, so per-sample cost amortises over the
  batch. This makes exec latency nearly independent of sample count but
  introduces a small approximation error (scale 2^30 / 2^40).
  BFV uses integer arithmetic with no SIMD packing across samples; each
  sample occupies its own ciphertext slot and arithmetic is exact (MAE ≈ 0).
  The practical consequence: BFV total wall-clock time is lower (no SIMD
  overhead) but requires more storage per sample due to non-packed ciphertexts.

Storage / performance implications for encrypted genomics pipelines:
  Storing encrypted RNA-Seq data at N=16384 costs ~835 KB per sample (BFV) or
  ~264 KB per sample (CKKS). For a typical cohort of 800 samples this yields
  ~668 MB (BFV) or ~211 MB (CKKS) of ciphertext storage — roughly 10-40× the
  plaintext footprint. Pipeline designers must account for this when sizing
  object storage (e.g. S3/GCS) and network bandwidth for encrypted data
  transfer. CKKS is the better choice when storage budget is constrained and
  small approximation error (~10^-5) is acceptable; BFV is preferable when
  exact integer arithmetic is required and storage cost is secondary.
"""

import matplotlib.ticker as ticker

# ── Color palette ──────────────────────────────────────────────────────────
# CKKS: warm reds/oranges   BFV: cool blues
COLORS = {
    ('CKKS', 16384): '#C0392B',
    ('CKKS',  8192): '#E67E22',
    ('BFV',  16384): '#1A6EA8',
    ('BFV',   8192): '#5DADE2',
    ('BFV',   4096): '#AED6F1',
}

# ── Markers ────────────────────────────────────────────────────────────────
MARKERS = {
    ('CKKS', 16384): 'o',
    ('CKKS',  8192): 'o',
    ('BFV',  16384): 's',
    ('BFV',   8192): 's',
    ('BFV',   4096): 's',
}

# ── Line styles ────────────────────────────────────────────────────────────
LINESTYLES = {
    'CKKS': '-',
    'BFV':  '--',
}

# ── Line/marker sizes ──────────────────────────────────────────────────────
LW = 2.2
MS = 6

# ── Band alpha ─────────────────────────────────────────────────────────────
BAND_ALPHA = 0.12

# ── Grid / spine style (apply to every ax) ────────────────────────────────
def style_ax(ax):
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.grid(True, which='major', linestyle=':', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)

# ── Legend helper ──────────────────────────────────────────────────────────
LEGEND_KW = dict(framealpha=0.92, edgecolor='#cccccc',
                 handlelength=2, labelspacing=0.4)

def scheme_label(scheme, pmd):
    return f'{scheme} (N={pmd})'

# ── Environment footnote (added to every figure) ──────────────────────────
ENV_NOTE = (
    "Env: Intel i7-10750H, 16 GB RAM, Ubuntu 22.04 (WSL2), "
    "Python 3.12, TenSEAL 0.3.14  |  Results averaged across 10 runs ± 1 std"
)

def add_env_note(fig):
    """Add hardware/software footnote at bottom of every figure."""
    fig.text(0.5, -0.02, ENV_NOTE, ha='center', va='top',
             fontsize=7, color='#555555', style='italic')

# ── Log-scale axis label helpers ──────────────────────────────────────────
def log10_formatter():
    """Formatter for log10 y-axes: shows 10^x notation."""
    return ticker.FuncFormatter(lambda x, _: f'$10^{{{int(round(import_math_log10(x)))}}}$'
                                if x > 0 else '0')

def log2_ylabel(unit='KB'):
    return f'Ciphertext size ({unit}, log₂ scale)'

def log10_ylabel(unit='ms'):
    return f'Latency ({unit}, log₁₀ scale)'

# helper used inside formatter above — avoids circular import
import math as _math
def import_math_log10(x):
    return _math.log10(x) if x > 0 else 0
