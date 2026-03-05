"""
BFV Validation — Phase 2C
Operations  : Group sum accumulation + ciphertext subtraction (DE scoring)
Depth cost  : 0 multiplications — additions and subtraction only
Compatible  : 8192, 16384  (4096 expected excluded — insufficient slots)

Mirrors program1/2/3 CKKS pilot structure exactly.
Automates across poly_mod_degree = 4096, 8192, 16384 with 6 runs each.

Scaling convention (fixed — cross-scheme comparability with CKKS):
  encode : round(de_score * SCALE_FACTOR) -> int
  decode : int / SCALE_FACTOR             -> float
  SCALE_FACTOR = 10_000
  Max DE ~0.90 -> max int 9_000
  Max col sum 50*9_000 = 450_000 < plain_modulus 786_433

Output: results/bfv_validation_results.csv  (columns mirror pilot_results.csv)
"""

import seal
from seal import (
    EncryptionParameters, scheme_type, SEALContext,
    KeyGenerator, Encryptor, Evaluator, Decryptor,
    BatchEncoder, Plaintext, Ciphertext, CoeffModulus,
)
import numpy as np
import pandas as pd
import os, time, tempfile

# ==========================================================
# CONFIGURATION
# ==========================================================
SCALE_FACTOR     = 10_000
N_FEATURES       = 500        # top-variance features, fixed from Phase 2
N_SAMPLES_TEST   = 100        # synthetic batch matching batch_a
RANDOM_STATE     = 42
REPEATS          = 6          # matches NumRuns in pilot_results.csv
MAE_THRESHOLD    = 1.0 / SCALE_FACTOR   # 1e-4 — rounding noise only

# plain_modulus = 786_433  (prime, 3*2^18+1, NTT-friendly)
# Max intermediate col sum = 50 * 9_000 = 450_000 < 786_433 — safe
PLAIN_MODULUS    = 786433
POLY_MOD_DEGREES = [4096, 8192, 16384]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ==========================================================
# CONTEXT
# ==========================================================
def create_context(poly_mod_degree, plain_modulus):
    try:
        parms = EncryptionParameters(scheme_type.bfv)
        parms.set_poly_modulus_degree(poly_mod_degree)
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_mod_degree))
        parms.set_plain_modulus(plain_modulus)
        ctx = SEALContext(parms)
        if not ctx.parameters_set():
            return None, None
        return ctx, BatchEncoder(ctx)
    except Exception:
        return None, None

# ==========================================================
# SYNTHETIC RNA-SEQ-LIKE DATA
# Two cancer groups, floats in [0,1] after min-max norm.
# Encoded as int64 via SCALE_FACTOR — mirrors Phase 2 preprocessing.
# ==========================================================
def make_data(n_samples, n_features, seed):
    rng   = np.random.default_rng(seed)
    n_a   = n_samples // 2
    n_b   = n_samples - n_a
    grp_a = rng.uniform(0.0, 1.0, (n_a, n_features))
    grp_b = rng.uniform(0.0, 1.0, (n_b, n_features))
    pt_de = np.abs(grp_a.mean(axis=0) - grp_b.mean(axis=0))
    a_enc = np.round(grp_a * SCALE_FACTOR).astype(np.int64)
    b_enc = np.round(grp_b * SCALE_FACTOR).astype(np.int64)
    return a_enc, b_enc, pt_de, n_a

def max_col_sum(a_enc, b_enc):
    return max(int(a_enc.sum(axis=0).max()), int(b_enc.sum(axis=0).max()))

def ct_size_kb(ct):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        fname = f.name
    ct.save(fname)
    kb = os.path.getsize(fname) / 1024.0
    os.unlink(fname)
    return kb

# ==========================================================
# BENCHMARK LOOP — one poly_mod_degree at a time
# Mirrors the enc / exec / dec benchmark structure of program1
# ==========================================================
def run_config(poly_mod_degree):

    print(f"\n========== BFV VALIDATION: poly_mod_degree={poly_mod_degree} ==========")
    print(f"Plain modulus             : {PLAIN_MODULUS:,}")
    print(f"Vector size (features)    : {N_FEATURES}")
    print(f"Samples (synthetic batch) : {N_SAMPLES_TEST}")
    print(f"Repeats per measurement   : {REPEATS}")

    row = dict(
        Config=f"BFV-{poly_mod_degree}", PolyModDegree=poly_mod_degree,
        PlainModulus=PLAIN_MODULUS, VectorSize=N_FEATURES,
        SlotCapacityOK=False, OverflowOK=False,
        EncLatency_mean_ms=None, EncLatency_std_ms=None,
        ExecLatency_mean_ms=None, ExecLatency_std_ms=None,
        DecLatency_mean_ms=None,  DecLatency_std_ms=None,
        CiphertextSize_mean_KB=None, CiphertextSize_std_KB=None,
        MAE_mean=None, MAE_std=None,
        NumRuns=REPEATS, Valid=False, Notes="",
    )

    ctx, encoder = create_context(poly_mod_degree, PLAIN_MODULUS)
    if ctx is None:
        row["Notes"] = "SEAL rejected parameters"
        print("------------------------------------------------")
        print("Result                    : EXCLUDED (SEAL rejected parameters)")
        print("=" * 60)
        return row

    n_slots = encoder.slot_count()
    print(f"Slots available           : {n_slots:,}")

    # Slot capacity check — need n_slots >= N_FEATURES to fit one sample per CT
    if n_slots < N_FEATURES:
        row["Notes"] = f"EXCLUDED -- {n_slots} slots < {N_FEATURES} features required"
        print("------------------------------------------------")
        print(f"Slot capacity             : FAIL ({n_slots} < {N_FEATURES})")
        print("Result                    : EXCLUDED")
        print("=" * 60)
        return row
    row["SlotCapacityOK"] = True
    print(f"Slot capacity             : OK ({n_slots} >= {N_FEATURES})")

    # Overflow check — worst-case intermediate sum must stay < plain_modulus
    a0, b0, _, _ = make_data(N_SAMPLES_TEST, N_FEATURES, RANDOM_STATE)
    mx = max_col_sum(a0, b0)
    row["OverflowOK"] = mx < PLAIN_MODULUS
    print(f"Max intermediate sum      : {mx:,} (plain_modulus={PLAIN_MODULUS:,}) -> {'OK' if row['OverflowOK'] else 'OVERFLOW'}")
    if not row["OverflowOK"]:
        row["Notes"] = f"EXCLUDED -- overflow: max sum {mx:,} >= {PLAIN_MODULUS:,}"
        print("Result                    : EXCLUDED")
        print("=" * 60)
        return row

    def enc_row(enc, row_vec):
        padded = list(row_vec) + [0] * (n_slots - len(row_vec))
        pt = Plaintext()
        encoder.encode(padded, pt)
        ct = Ciphertext()
        enc.encrypt(pt, ct)
        return ct

    # Warm-up — first call always slower due to JIT/cache effects
    a_w, b_w, _, _ = make_data(N_SAMPLES_TEST, N_FEATURES, RANDOM_STATE)
    kg_w = KeyGenerator(ctx)
    enc_w = Encryptor(ctx, kg_w.create_public_key())
    _ = [enc_row(enc_w, r) for r in a_w]

    # ==========================================================
    # ENCRYPTION BENCHMARK
    # Time each full batch encryption independently
    # ==========================================================
    enc_times = []
    for i in range(REPEATS):
        a_i, b_i, _, _ = make_data(N_SAMPLES_TEST, N_FEATURES, RANDOM_STATE + i + 1)
        kg  = KeyGenerator(ctx)
        enc = Encryptor(ctx, kg.create_public_key())
        start = time.perf_counter()
        ca = [enc_row(enc, r) for r in a_i]
        cb = [enc_row(enc, r) for r in b_i]
        enc_times.append((time.perf_counter() - start) * 1000)

    # ==========================================================
    # EXECUTION BENCHMARK
    # Re-encrypt each iteration for consistent ciphertext state
    # ==========================================================
    exec_times = []
    for i in range(REPEATS):
        a_i, b_i, _, _ = make_data(N_SAMPLES_TEST, N_FEATURES, RANDOM_STATE + i + 1)
        kg  = KeyGenerator(ctx)
        enc = Encryptor(ctx, kg.create_public_key())
        ev  = Evaluator(ctx)
        ca  = [enc_row(enc, r) for r in a_i]
        cb  = [enc_row(enc, r) for r in b_i]
        start = time.perf_counter()
        sa = ca[0]
        for c in ca[1:]: ev.add_inplace(sa, c)
        sb = cb[0]
        for c in cb[1:]: ev.add_inplace(sb, c)
        cd = Ciphertext()
        ev.sub(sa, sb, cd)
        exec_times.append((time.perf_counter() - start) * 1000)

    # ==========================================================
    # DECRYPTION BENCHMARK
    # Compute result once outside loop — benchmark decrypt only
    # ==========================================================
    a_d, b_d, pt_de, n_a = make_data(N_SAMPLES_TEST, N_FEATURES, RANDOM_STATE + 1)
    kg  = KeyGenerator(ctx)
    pk  = kg.create_public_key()
    sk  = kg.secret_key()
    enc = Encryptor(ctx, pk)
    ev  = Evaluator(ctx)
    dec = Decryptor(ctx, sk)
    ca  = [enc_row(enc, r) for r in a_d]
    cb  = [enc_row(enc, r) for r in b_d]
    sa  = ca[0]
    for c in ca[1:]: ev.add_inplace(sa, c)
    sb  = cb[0]
    for c in cb[1:]: ev.add_inplace(sb, c)
    result = Ciphertext()
    ev.sub(sa, sb, result)

    dec_times = []
    for _ in range(REPEATS):
        pt_out = Plaintext()
        start  = time.perf_counter()
        dec.decrypt(result, pt_out)
        dec_times.append((time.perf_counter() - start) * 1000)

    # Final decode for MAE and ciphertext size
    decoded = encoder.decode_int64(pt_out)
    diff    = np.array(decoded[:N_FEATURES], dtype=np.float64)
    enc_de  = np.abs(diff / (n_a * SCALE_FACTOR))
    mae     = float(np.mean(np.abs(enc_de - pt_de)))
    size_kb = ct_size_kb(result)

    # ==========================================================
    # OUTPUT  (mirrors program1 print block exactly)
    # ==========================================================
    print("------------------------------------------------")
    print(f"Encryption latency  (ms)  : {np.mean(enc_times):.4f}  +- {np.std(enc_times):.4f}")
    print(f"Execution latency   (ms)  : {np.mean(exec_times):.4f}  +- {np.std(exec_times):.4f}")
    print(f"Decryption latency  (ms)  : {np.mean(dec_times):.4f}  +- {np.std(dec_times):.4f}")
    print(f"Ciphertext size     (KB)  : {size_kb:.4f}")
    print(f"MAE                       : {mae:.10f}")
    print(f"Result                    : {'VALID' if mae < MAE_THRESHOLD else 'FAIL -- MAE above threshold'}")
    print("=" * 60)

    row.update(
        EncLatency_mean_ms=float(np.mean(enc_times)),    EncLatency_std_ms=float(np.std(enc_times)),
        ExecLatency_mean_ms=float(np.mean(exec_times)),  ExecLatency_std_ms=float(np.std(exec_times)),
        DecLatency_mean_ms=float(np.mean(dec_times)),    DecLatency_std_ms=float(np.std(dec_times)),
        CiphertextSize_mean_KB=size_kb, CiphertextSize_std_KB=0.0,
        MAE_mean=mae, MAE_std=0.0,
    )
    row["Valid"] = mae < MAE_THRESHOLD
    row["Notes"] = "VALID" if row["Valid"] else f"FAIL -- MAE {mae:.4e} >= threshold {MAE_THRESHOLD:.4e}"
    return row

# ==========================================================
# MAIN — loop over all poly_mod_degrees, save CSV
# ==========================================================
print("\n" + "=" * 60)
print("  BFV PARAMETER VALIDATION  --  Phase 2C")
print("  CKKS vs BFV  --  RNA-Seq DE Scoring")
print("=" * 60)

rows = [run_config(pmd) for pmd in POLY_MOD_DEGREES]

print("\n========== SUMMARY ==========")
print(f"{'Config':<16} {'SlotOK':>7} {'OvflOK':>7} {'MAE':>12} {'Valid':>10}")
print("-" * 56)
for r in rows:
    mae_s = f"{r['MAE_mean']:.4e}" if r["MAE_mean"] is not None else "--"
    print(f"{r['Config']:<16} {'YES' if r['SlotCapacityOK'] else 'NO':>7} {'YES' if r['OverflowOK'] else 'NO':>7} {mae_s:>12} {'VALID' if r['Valid'] else 'EXCLUDED':>10}")
print("=" * 56)

col_order = [
    "Config", "PolyModDegree", "PlainModulus", "VectorSize",
    "SlotCapacityOK", "OverflowOK",
    "EncLatency_mean_ms", "EncLatency_std_ms",
    "ExecLatency_mean_ms", "ExecLatency_std_ms",
    "DecLatency_mean_ms", "DecLatency_std_ms",
    "CiphertextSize_mean_KB", "CiphertextSize_std_KB",
    "MAE_mean", "MAE_std",
    "NumRuns", "Valid", "Notes",
]
out = os.path.join(RESULTS_DIR, "bfv_validation_results.csv")
pd.DataFrame(rows)[col_order].to_csv(out, index=False)
print(f"\nSaved: {out}")
print("Next: paste bfv_validation_results.csv into Claude, then build phase3_bfv_de.py")