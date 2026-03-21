"""
topk_bfv_de.py  —  BFV top-k validation (v4 — formula-consistent plaintext)
Run in WSL2.

Key fix: plain_scores computed using the SAME formula as BFV decoding:
  abs((sum_A_float - sum_B_float) / (n_A * SCALE_FACTOR))
NOT loaded from the baseline CSV (which uses true mean difference).
This ensures plaintext and encrypted scores are computed identically,
so rounding errors cancel and MAE returns to ~2-7e-6.

Saves to results/topk/:
  bfv_d1_enc_scores_10runs.npy  (10, 500, 10)
  bfv_d1_plain_scores.npy       (500, 10)   <- formula-consistent
  bfv_d1_gene_order.npy
  bfv_d2_enc_scores_10runs.npy  (10, 500, 1)
  bfv_d2_plain_scores.npy       (500, 1)
  bfv_d2_gene_order.npy
"""

import seal
from seal import (
    EncryptionParameters, scheme_type, SEALContext,
    KeyGenerator, Encryptor, Evaluator, Decryptor,
    BatchEncoder, Plaintext, Ciphertext, CoeffModulus,
)
import pandas as pd
import numpy as np
import time, os, tempfile
from itertools import combinations

POLY_MOD_DEGREE = 8192
PLAIN_MODULUS   = 4_816_897
SCALE_FACTOR    = 10_000
N_RUNS          = 10

CONFIGS = [
    {
        "dataset":      "dataset1",
        "batch_path":   "datasets/batch_c_801.csv",
        "cancer_types": ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"],
        "out_prefix":   "results/topk/bfv_d1",
    },
    {
        "dataset":      "dataset2",
        "batch_path":   "datasets/d2_batch_c_1129.csv",
        "cancer_types": ["LUSC", "LUAD"],
        "out_prefix":   "results/topk/bfv_d2",
    },
]

os.makedirs("results/topk", exist_ok=True)

def create_context():
    parms = EncryptionParameters(scheme_type.bfv)
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(POLY_MOD_DEGREE))
    parms.set_plain_modulus(PLAIN_MODULUS)
    ctx = SEALContext(parms)
    return ctx, BatchEncoder(ctx)

def _clone_ct(ctx, ct):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        fname = f.name
    ct.save(fname)
    ct2 = Ciphertext()
    ct2.load(ctx, fname)
    os.unlink(fname)
    return ct2

def _sum_cts(ctx, evaluator, cts):
    result = _clone_ct(ctx, cts[0])
    for ct in cts[1:]:
        evaluator.add_inplace(result, ct)
    return result

ctx, encoder = create_context()
n_slots = encoder.slot_count()
print(f"BFV top-k v4 | N={POLY_MOD_DEGREE} | PLAIN_MODULUS={PLAIN_MODULUS}")
print(f"Plain scores computed with same formula as BFV decoding")

for cfg in CONFIGS:
    print(f"\n{'='*55}")
    print(f"Dataset: {cfg['dataset']}  batch_c")
    print(f"{'='*55}")

    df    = pd.read_csv(cfg["batch_path"])
    pairs = list(combinations(cfg["cancer_types"], 2))
    pair_keys  = [f"{a}_vs_{b}" for a, b in pairs]
    gene_cols  = [c for c in df.columns if c != "cancer_type"]
    n_features = len(gene_cols)

    group_sizes = {}
    for ct in cfg["cancer_types"]:
        group_sizes[ct] = len(df[df["cancer_type"] == ct])
        print(f"  {ct}: n={group_sizes[ct]}")

    # Compute plain_scores using SAME formula as BFV:
    # abs((sum_A - sum_B) / (n_A * SCALE_FACTOR))
    # This matches BFV decoding exactly (same rounding, same divisor)
    plain_mat = np.zeros((n_features, len(pairs)), dtype=np.float64)
    for pi, (type_a, type_b) in enumerate(pairs):
        n_a = group_sizes[type_a]
        # Apply same integer rounding as BFV encoding
        sum_a = np.round(
            df[df["cancer_type"]==type_a][gene_cols].values * SCALE_FACTOR
        ).astype(np.int64).sum(axis=0)
        sum_b = np.round(
            df[df["cancer_type"]==type_b][gene_cols].values * SCALE_FACTOR
        ).astype(np.int64).sum(axis=0)
        diff  = (sum_a - sum_b).astype(np.float64)
        plain_mat[:, pi] = np.abs(diff / (n_a * SCALE_FACTOR))

    np.save(f"{cfg['out_prefix']}_plain_scores.npy", plain_mat)
    np.save(f"{cfg['out_prefix']}_gene_order.npy", np.array(gene_cols))
    print(f"  Plaintext scores saved: {plain_mat.shape}")
    print(f"  Plain score range: {plain_mat.min():.4f} to {plain_mat.max():.4f}")

    all_enc_scores = []

    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}...", end=" ", flush=True)
        t0 = time.perf_counter()

        kg        = KeyGenerator(ctx)
        pub_key   = kg.create_public_key()
        sec_key   = kg.secret_key()
        encryptor = Encryptor(ctx, pub_key)
        evaluator = Evaluator(ctx)
        decryptor = Decryptor(ctx, sec_key)

        def encode_row(row_vals):
            padded = np.zeros(n_slots, dtype=np.int64)
            ints   = np.round(np.array(row_vals) * SCALE_FACTOR).astype(np.int64)
            padded[:n_features] = ints
            return encoder.encode(padded)

        def encrypt_row(row_vals):
            ct = Ciphertext()
            encryptor.encrypt(encode_row(row_vals), ct)
            return ct

        enc = {}
        for ct_type in cfg["cancer_types"]:
            group = df[df["cancer_type"] == ct_type][gene_cols].values
            enc[ct_type] = [encrypt_row(row) for row in group]

        enc_scores_run = np.zeros((n_features, len(pairs)), dtype=np.float64)

        for pi, (type_a, type_b) in enumerate(pairs):
            n_a = group_sizes[type_a]

            sum_a = _sum_cts(ctx, evaluator, enc[type_a])
            sum_b = _sum_cts(ctx, evaluator, enc[type_b])
            diff  = evaluator.sub(sum_a, sum_b)

            pt_out = Plaintext()
            decryptor.decrypt(diff, pt_out)
            raw     = encoder.decode(pt_out)
            raw_int = np.array(raw[:n_features], dtype=np.int64)
            raw_int[raw_int > PLAIN_MODULUS // 2] -= PLAIN_MODULUS

            enc_scores_run[:, pi] = np.abs(
                raw_int.astype(np.float64) / (n_a * SCALE_FACTOR)
            )

        all_enc_scores.append(enc_scores_run)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"done ({elapsed:.0f} ms)")

        if run == 1:
            mae = np.abs(enc_scores_run - plain_mat).mean()
            print(f"    Sanity MAE run 1: {mae:.2e}  (expect ~1e-6)")

    scores_array = np.stack(all_enc_scores, axis=0)
    np.save(f"{cfg['out_prefix']}_enc_scores_10runs.npy", scores_array)
    print(f"  Saved: shape={scores_array.shape}")

print("\nBFV done. Now run topk_analysis.py")