"""
topk_ckks_de.py  —  CKKS top-k validation (FIXED)
Run in Windows PowerShell with ckks_env active.

Key fix vs previous version:
  - Decrypts actual encrypted scores (no synthetic noise)
  - Gene order locked to baseline CSV order
  - Saves real per-gene decrypted vectors

Saves to results/topk/:
  ckks_d1_enc_scores_10runs.npy  (10, n_genes, n_pairs)
  ckks_d1_plain_scores.npy       (n_genes, n_pairs)
  ckks_d1_gene_order.npy
  ckks_d2_enc_scores_10runs.npy  (10, n_genes, 1)
  ckks_d2_plain_scores.npy       (n_genes, 1)
  ckks_d2_gene_order.npy
"""

import tenseal as ts
import pandas as pd
import numpy as np
import time, os
from itertools import combinations

POLY_MOD_DEGREE = 8192
COEFF_MOD_BITS  = [40, 30, 30, 40]
SCALE           = 2 ** 30
N_RUNS          = 10

CONFIGS = [
    {
        "dataset":      "dataset1",
        "batch_path":   "datasets/batch_c_801.csv",
        "baseline":     "scoring/dataset1/de_baselines/de_baseline_batch_c.csv",
        "cancer_types": ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"],
        "out_prefix":   "results/topk/ckks_d1",
    },
    {
        "dataset":      "dataset2",
        "batch_path":   "datasets/d2_batch_c_1129.csv",
        "baseline":     "scoring/dataset2/d2_de_baseline_batch_c.csv",
        "cancer_types": ["LUSC", "LUAD"],
        "out_prefix":   "results/topk/ckks_d2",
    },
]

os.makedirs("results/topk", exist_ok=True)

def make_ctx():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BITS,
    )
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    return ctx

print(f"CKKS top-k | N={POLY_MOD_DEGREE} | scale=2^30")

for cfg in CONFIGS:
    print(f"\n{'='*55}")
    print(f"Dataset: {cfg['dataset']}  batch_c")
    print(f"{'='*55}")

    df       = pd.read_csv(cfg["batch_path"])
    baseline = pd.read_csv(cfg["baseline"], index_col=0)
    pairs    = list(combinations(cfg["cancer_types"], 2))
    pair_keys = [f"{a}_vs_{b}" for a, b in pairs]

    # Lock gene order to baseline
    gene_cols  = list(baseline.index)
    n_features = len(gene_cols)

    group_sizes = {}
    for ct in cfg["cancer_types"]:
        group_sizes[ct] = len(df[df["cancer_type"] == ct])
        print(f"  {ct}: n={group_sizes[ct]}")

    plain_mat = baseline[pair_keys].values   # (n_genes, n_pairs)
    np.save(f"{cfg['out_prefix']}_plain_scores.npy", plain_mat)
    np.save(f"{cfg['out_prefix']}_gene_order.npy", np.array(gene_cols))
    print(f"  Plaintext scores saved: {plain_mat.shape}")

    all_enc_scores = []

    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}...", end=" ", flush=True)
        t0  = time.perf_counter()
        ctx = make_ctx()

        enc_scores_run = np.zeros((n_features, len(pairs)), dtype=np.float64)

        for pi, (type_a, type_b) in enumerate(pairs):
            n_a = group_sizes[type_a]
            n_b = group_sizes[type_b]

            group_a = df[df["cancer_type"] == type_a][gene_cols].values
            group_b = df[df["cancer_type"] == type_b][gene_cols].values

            # Encrypt
            enc_a = [ts.ckks_vector(ctx, row.tolist()) for row in group_a]
            enc_b = [ts.ckks_vector(ctx, row.tolist()) for row in group_b]

            # Sum
            sum_a = enc_a[0]
            for v in enc_a[1:]:
                sum_a = sum_a + v
            sum_b = enc_b[0]
            for v in enc_b[1:]:
                sum_b = sum_b + v

            # Mean via scalar multiply
            mean_a = sum_a * (1.0 / n_a)
            mean_b = sum_b * (1.0 / n_b)

            # Subtract and decrypt — this is the actual encrypted result
            diff = mean_a - mean_b
            dec  = np.array(diff.decrypt()[:n_features])

            enc_scores_run[:, pi] = np.abs(dec)

        all_enc_scores.append(enc_scores_run)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"done ({elapsed:.0f} ms)")

        if run == 1:
            mae = np.abs(enc_scores_run - plain_mat).mean()
            print(f"    Sanity MAE run 1: {mae:.2e}  (expect ~1e-5 for CKKS)")

    scores_array = np.stack(all_enc_scores, axis=0)
    np.save(f"{cfg['out_prefix']}_enc_scores_10runs.npy", scores_array)
    print(f"  Saved: shape={scores_array.shape}")

print("\nCKKS done. Now run topk_analysis.py")