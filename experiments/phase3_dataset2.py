import tenseal as ts
import pandas as pd
import numpy as np
import time
import os
import csv
from tqdm import tqdm
from itertools import combinations

# ==========================================================
# CONFIGURATION
# ==========================================================
BATCHES = {
    "batch_a": ("datasets/d2_batch_a_100.csv",       "scoring/dataset2/d2_de_baseline_batch_a.csv"),
    "batch_b": ("datasets/d2_batch_b_400.csv",       "scoring/dataset2/d2_de_baseline_batch_b.csv"),
    "batch_c": ("datasets/d2_batch_c_1129.csv",      "scoring/dataset2/d2_de_baseline_batch_c.csv"),
}

MODULI       = [8192, 16384]
N_RUNS       = 10
SCALE        = 2**30
CANCER_TYPES = ["LUSC", "LUAD"]
PAIRS        = list(combinations(CANCER_TYPES, 2))

OUTPUT_DIR  = "results/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase3_ckks_dataset2.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELDNAMES = [
    "dataset", "scheme", "poly_mod_degree", "batch",
    "samples", "run", "enc_latency_ms", "exec_latency_ms",
    "dec_latency_ms", "ct_size_kb", "mae"
]

# ==========================================================
# CONTEXT
# ==========================================================
def create_context(poly_mod_degree):
    sizes = {
        8192:  [40, 30, 30, 40],
        16384: [40, 30, 30, 30, 30, 40]
    }
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=sizes[poly_mod_degree]
    )
    ctx.global_scale = SCALE
    ctx.generate_galois_keys()
    return ctx

# ==========================================================
# ENCRYPTED DE SCORING
# ==========================================================
def run_encrypted_de(ctx, df, cancer_types, pairs):
    gene_cols = [c for c in df.columns if c != "cancer_type"]

    t0 = time.perf_counter()
    enc = {}
    for ct in cancer_types:
        group = df[df["cancer_type"] == ct][gene_cols].values
        enc[ct] = [ts.ckks_vector(ctx, row.tolist()) for row in group]
    enc_latency = (time.perf_counter() - t0) * 1000

    ct_size_kb = len(enc[cancer_types[0]][0].serialize()) / 1024

    t0 = time.perf_counter()
    de_enc = {}
    for type_a, type_b in pairs:
        if not enc[type_a] or not enc[type_b]:
            continue
        sum_a = enc[type_a][0].copy()
        for v in enc[type_a][1:]: sum_a += v
        mean_a = sum_a * (1.0 / len(enc[type_a]))

        sum_b = enc[type_b][0].copy()
        for v in enc[type_b][1:]: sum_b += v
        mean_b = sum_b * (1.0 / len(enc[type_b]))

        de_enc[f"{type_a}_vs_{type_b}"] = mean_a - mean_b
    exec_latency = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    de_dec = {k: np.abs(np.array(v.decrypt())) for k, v in de_enc.items()}
    dec_latency = (time.perf_counter() - t0) * 1000

    return enc_latency, exec_latency, dec_latency, ct_size_kb, de_dec

# ==========================================================
# MAE
# ==========================================================
def compute_mae(de_dec, baseline_df):
    pair_cols = [c for c in baseline_df.columns if c != "gene"]
    errors = []
    for pair in pair_cols:
        if pair not in de_dec:
            continue
        pt  = baseline_df[pair].values
        enc = de_dec[pair][:len(pt)]
        errors.append(np.abs(pt - enc).mean())
    return np.mean(errors)

# ==========================================================
# MAIN
# ==========================================================
with open(OUTPUT_FILE, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

total_configs = len(MODULI) * len(BATCHES)
config_num    = 0

print("PHASE 3 — CKKS — DATASET 2 (LUSC + LUAD)")
print(f"Configs: {total_configs} | Runs per config: {N_RUNS} | Total runs: {total_configs * N_RUNS}")
print()

for poly_mod in MODULI:
    print(f"{'='*55}")
    print(f"poly_mod_degree = {poly_mod}")
    print(f"{'='*55}")

    ctx = create_context(poly_mod)

    for batch_name, (batch_path, baseline_path) in BATCHES.items():
        config_num += 1
        df       = pd.read_csv(batch_path)
        baseline = pd.read_csv(baseline_path)
        run_results = []

        print(f"\nConfig {config_num}/{total_configs} — {batch_name} ({len(df)} samples)")

        for run in tqdm(range(1, N_RUNS + 1), desc=f"  {batch_name} mod={poly_mod}", ncols=70):
            enc_lat, exec_lat, dec_lat, ct_kb, de_dec = run_encrypted_de(
                ctx, df, CANCER_TYPES, PAIRS
            )
            mae = compute_mae(de_dec, baseline)

            row = {
                "dataset":         "dataset2",
                "scheme":          "CKKS",
                "poly_mod_degree": poly_mod,
                "batch":           batch_name,
                "samples":         len(df),
                "run":             run,
                "enc_latency_ms":  round(enc_lat,  4),
                "exec_latency_ms": round(exec_lat, 4),
                "dec_latency_ms":  round(dec_lat,  4),
                "ct_size_kb":      round(ct_kb,    4),
                "mae":             round(mae,       8),
            }
            run_results.append(row)

            with open(OUTPUT_FILE, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        print(f"  Enc latency  : {np.mean([r['enc_latency_ms']  for r in run_results]):.2f} ms")
        print(f"  Exec latency : {np.mean([r['exec_latency_ms'] for r in run_results]):.2f} ms")
        print(f"  Dec latency  : {np.mean([r['dec_latency_ms']  for r in run_results]):.2f} ms")
        print(f"  CT size      : {np.mean([r['ct_size_kb']      for r in run_results]):.2f} KB")
        print(f"  MAE          : {np.mean([r['mae']             for r in run_results]):.8f}")

print(f"\nDONE — results saved to {OUTPUT_FILE}")