import seal
from seal import (
    EncryptionParameters, scheme_type, SEALContext,
    KeyGenerator, Encryptor, Evaluator, Decryptor,
    BatchEncoder, Plaintext, Ciphertext, CoeffModulus,
)
import pandas as pd
import numpy as np
import time
import os
import csv
import tempfile
from tqdm import tqdm
from itertools import combinations

# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET = "dataset"   # change to "dataset2" for D2 run

BATCHES_D1 = {
    "batch_a": ("datasets/batch_a_100.csv",         "scoring/dataset1/de_baselines/de_baseline_batch_a.csv"),
    "batch_b": ("datasets/batch_b_400.csv",          "scoring/dataset1/de_baselines/de_baseline_batch_b.csv"),
    "batch_c": ("datasets/batch_c_801.csv",          "scoring/dataset1/de_baselines/de_baseline_batch_c.csv"),
}

BATCHES_D2 = {
    "batch_a": ("datasets/d2_batch_a_100.csv",       "scoring/dataset2/d2_de_baseline_batch_a.csv"),
    "batch_b": ("datasets/d2_batch_b_400.csv",       "scoring/dataset2/d2_de_baseline_batch_b.csv"),
    "batch_c": ("datasets/d2_batch_c_1129.csv",      "scoring/dataset2/d2_de_baseline_batch_c.csv"),
}

BATCHES = BATCHES_D1 if DATASET == "dataset1" else BATCHES_D2

POLY_MOD_DEGREES = [4096, 8192, 16384]
PLAIN_MODULUS    = 4816897
SCALE_FACTOR     = 10_000
N_RUNS           = 10

CANCER_TYPES_D1 = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]
CANCER_TYPES_D2 = ["LUSC", "LUAD"]
CANCER_TYPES = CANCER_TYPES_D1 if DATASET == "dataset1" else CANCER_TYPES_D2
PAIRS        = list(combinations(CANCER_TYPES, 2))

OUTPUT_DIR  = "results/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"phase3_bfv_{DATASET}.csv")
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
    parms = EncryptionParameters(scheme_type.bfv)
    parms.set_poly_modulus_degree(poly_mod_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_mod_degree))
    parms.set_plain_modulus(PLAIN_MODULUS)
    ctx = SEALContext(parms)
    return ctx, BatchEncoder(ctx)

# ==========================================================
# HELPERS
# ==========================================================
def _ct_kb(ct):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        fname = f.name
    ct.save(fname)
    kb = os.path.getsize(fname) / 1024.0
    os.unlink(fname)
    return kb

def _clone_ct(ct, ctx):
    """Clone a ciphertext via save/load."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        fname = f.name
    ct.save(fname)
    ct2 = Ciphertext()
    ct2.load(ctx, fname)
    os.unlink(fname)
    return ct2

def _sum_cts(evaluator, cts, ctx):
    """Sum a list of ciphertexts without mutating the originals."""
    result = _clone_ct(cts[0], ctx)
    for ct in cts[1:]:
        result = evaluator.add(result, ct)
    return result

# ==========================================================
# ENCRYPTED DE SCORING
# ==========================================================
def run_encrypted_de(ctx, encoder, df, cancer_types, pairs):
    gene_cols  = [c for c in df.columns if c != "cancer_type"]
    n_slots    = encoder.slot_count()
    n_features = len(gene_cols)

    kg        = KeyGenerator(ctx)
    pub_key   = kg.create_public_key()
    sec_key   = kg.secret_key()
    encryptor = Encryptor(ctx, pub_key)
    evaluator = Evaluator(ctx)
    decryptor = Decryptor(ctx, sec_key)

    def encode_row(row_vals):
        padded = np.zeros(n_slots, dtype=np.int64)
        ints   = np.round(np.array(row_vals) * SCALE_FACTOR).astype(np.int64)
        padded[:len(ints)] = ints
        return encoder.encode(padded)      # returns Plaintext directly

    def encrypt_pt(pt):
        ct = Ciphertext()
        encryptor.encrypt(pt, ct)
        return ct

    # -- Encryption ----------------------------------------------------------
    t0  = time.perf_counter()
    enc = {}
    for ct_type in cancer_types:
        group = df[df["cancer_type"] == ct_type][gene_cols].values
        enc[ct_type] = [encrypt_pt(encode_row(row)) for row in group]
    enc_latency = (time.perf_counter() - t0) * 1000

    ct_size_kb = _ct_kb(enc[cancer_types[0]][0])

    # -- Execution -----------------------------------------------------------
    t0     = time.perf_counter()
    de_enc = {}
    n_as   = {}
    for type_a, type_b in pairs:
        if not enc[type_a] or not enc[type_b]:
            continue
        sum_a = _sum_cts(evaluator, enc[type_a], ctx)
        sum_b = _sum_cts(evaluator, enc[type_b], ctx)
        diff  = evaluator.sub(sum_a, sum_b)
        key         = f"{type_a}_vs_{type_b}"
        de_enc[key] = diff
        n_as[key]   = len(enc[type_a])
    exec_latency = (time.perf_counter() - t0) * 1000

    # -- Decryption ----------------------------------------------------------
    t0     = time.perf_counter()
    de_dec = {}
    for key, ct in de_enc.items():
        pt_out = Plaintext()
        decryptor.decrypt(ct, pt_out)
        raw         = encoder.decode(pt_out)    # numpy array
        raw_int     = np.array(raw[:n_features], dtype=np.int64)
        de_dec[key] = np.abs(raw_int.astype(np.float64) / (n_as[key] * SCALE_FACTOR))
    dec_latency = (time.perf_counter() - t0) * 1000

    return enc_latency, exec_latency, dec_latency, ct_size_kb, de_dec

# ==========================================================
# MAE
# ==========================================================
def compute_mae(de_dec, baseline_df):
    pair_cols = [c for c in baseline_df.columns if c != "gene"]
    errors    = []
    for pair in pair_cols:
        if pair not in de_dec:
            continue
        pt  = baseline_df[pair].values
        enc = de_dec[pair][:len(pt)]
        errors.append(np.abs(pt - enc).mean())
    return np.mean(errors) if errors else float("nan")

# ==========================================================
# MAIN
# ==========================================================
with open(OUTPUT_FILE, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

total_configs = len(POLY_MOD_DEGREES) * len(BATCHES)
config_num    = 0

print(f"PHASE 3B — BFV — {DATASET.upper()}")
print(f"Configs: {total_configs} | Runs per config: {N_RUNS} | Total runs: {total_configs * N_RUNS}")
print(f"plain_modulus={PLAIN_MODULUS} | SCALE_FACTOR={SCALE_FACTOR}")
print(f"Cancer pairs: {len(PAIRS)} | {PAIRS}")
print()

for poly_mod in POLY_MOD_DEGREES:
    print(f"{'='*55}")
    print(f"poly_mod_degree = {poly_mod}")
    print(f"{'='*55}")

    ctx, encoder = create_context(poly_mod)
    n_slots      = encoder.slot_count()
    print(f"Slots: {n_slots} | Features needed: up to 500")

    for batch_name, (batch_path, baseline_path) in BATCHES.items():
        config_num += 1
        df          = pd.read_csv(batch_path)
        baseline    = pd.read_csv(baseline_path)
        n_features  = len([c for c in df.columns if c != "cancer_type"])
        run_results = []

        print(f"\nConfig {config_num}/{total_configs} — {batch_name} ({len(df)} samples, {n_features} features)")

        gene_cols = [c for c in df.columns if c != "cancer_type"]
        max_sum = max(
            int(np.round(df[df["cancer_type"] == ct][gene_cols].values * SCALE_FACTOR).sum(axis=0).max())
            for ct in CANCER_TYPES
        )
        if max_sum >= PLAIN_MODULUS:
            print(f"  WARNING: overflow — max col sum {max_sum:,} >= plain_modulus {PLAIN_MODULUS:,}")
        else:
            print(f"  Overflow check OK — max col sum {max_sum:,} < {PLAIN_MODULUS:,}")

        for run in tqdm(range(1, N_RUNS + 1), desc=f"  {batch_name} mod={poly_mod}", ncols=70):
            enc_lat, exec_lat, dec_lat, ct_kb, de_dec = run_encrypted_de(
                ctx, encoder, df, CANCER_TYPES, PAIRS
            )
            mae = compute_mae(de_dec, baseline)

            row = {
                "dataset":         DATASET,
                "scheme":          "BFV",
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
