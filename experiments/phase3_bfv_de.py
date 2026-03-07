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
DATASET = "dataset1"   # change to "dataset2" for D2 run

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
PLAIN_MODULUS    = 9338881   # prime, NTT-friendly for 4096/8192/16384; must be > 2*max_col_sum (9,304,726)
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
        return encoder.encode(padded)      # int64 array -- matches seal-python API

    def encrypt_pt(pt):
        ct = Ciphertext()
        encryptor.encrypt(pt, ct)
        return ct

    # -- Encryption (streaming) + Execution ----------------------------------
    # Encrypt row-by-row and accumulate immediately to avoid storing all
    # ciphertexts in RAM (prevents OOM kill at large batches / high PMD).
    # Timing is split correctly to match CKKS script:
    #   enc_latency  = pure encode + encrypt time only
    #   exec_latency = homomorphic add accumulation + final subtraction
    enc_sums    = {}
    n_per       = {}
    first_ct_kb = None
    enc_elapsed  = 0.0   # accumulate pure encryption time
    exec_elapsed = 0.0   # accumulate homomorphic add time

    for ct_type in cancer_types:
        group = df[df["cancer_type"] == ct_type][gene_cols].values
        n_per[ct_type] = len(group)
        running_sum = None
        for row in group:
            # -- timed: encode + encrypt only
            t0 = time.perf_counter()
            ct = encrypt_pt(encode_row(row))
            enc_elapsed += (time.perf_counter() - t0) * 1000

            if first_ct_kb is None:
                first_ct_kb = _ct_kb(ct)

            # -- timed: homomorphic addition
            # First row: running_sum is uninitialised — use ct directly.
            # No clone needed; ct is local and won't be reused.
            t0 = time.perf_counter()
            if running_sum is None:
                running_sum = ct
            else:
                running_sum = evaluator.add(running_sum, ct)
            exec_elapsed += (time.perf_counter() - t0) * 1000

        enc_sums[ct_type] = running_sum

    enc_latency = enc_elapsed

    exec_latency = exec_elapsed
    ct_size_kb   = first_ct_kb

    # -- Decryption ----------------------------------------------------------
    # Decrypt each group sum individually, then compute mean_A - mean_B in plaintext.
    # This matches CKKS exactly: CKKS computes encrypted(sum/n) then decrypts.
    # BFV can't do plaintext scalar divide in encrypted domain (integer-only modulus),
    # so we decrypt the sum and divide by n in plaintext -- mathematically identical.
    t0      = time.perf_counter()
    means   = {}
    for ct_type in cancer_types:
        if enc_sums.get(ct_type) is None:
            continue
        pt_out  = Plaintext()
        decryptor.decrypt(enc_sums[ct_type], pt_out)
        raw     = encoder.decode(pt_out)
        raw_int = np.array(raw[:n_features], dtype=np.int64)
        # Signed correction: BFV values are mod plain_modulus
        # Negative results wrap to [plain_modulus//2, plain_modulus)
        raw_int[raw_int > PLAIN_MODULUS // 2] -= PLAIN_MODULUS
        means[ct_type] = raw_int.astype(np.float64) / (n_per[ct_type] * SCALE_FACTOR)
    dec_latency = (time.perf_counter() - t0) * 1000

    de_dec = {}
    for type_a, type_b in pairs:
        if type_a not in means or type_b not in means:
            continue
        key        = f"{type_a}_vs_{type_b}"
        de_dec[key] = np.abs(means[type_a] - means[type_b])

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
