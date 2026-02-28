"""
Program 3 — High Complexity
Operations : Z-score normalization (mean subtraction + variance scaling)
             followed by an encrypted dot product with weight vector
Depth cost : 2 (one mul for variance scaling, one for dot product)
Compatible  : 8192, 16384
Note        : 4096 is depth-1 only — this program will raise a ValueError
              at 4096 by design. This is a documented finding: 4096 is
              insufficient for composite operations in this pipeline.
"""

import numpy as np
import time
import tenseal as ts
from ckks_utils import create_context

# ==========================================================
# CONFIGURATION — change POLY_MOD_DEGREE to test each level
# NOTE: 4096 will intentionally fail — document this as a finding
# ==========================================================
POLY_MOD_DEGREE = 16384
VECTOR_SIZE     = 4096
REPEATS         = 20

np.random.seed(42)

# ==========================================================
# DEPTH CHECK
# ==========================================================
if POLY_MOD_DEGREE == 4096:
    raise ValueError(
        "POLY_MOD_DEGREE=4096 does not support depth-2 operations. "
        "This is an expected finding: 4096 is insufficient for z-score "
        "normalization + dot product. Use 8192 or 16384."
    )

# ==========================================================
# CONTEXT
# ==========================================================
context = create_context(POLY_MOD_DEGREE)

# ==========================================================
# SYNTHETIC RNA-LIKE DATA
# 4096-element vector simulates a realistic gene expression sample
# after top-500 feature selection has been applied to the full dataset
# ==========================================================
vector       = np.abs(np.random.normal(loc=50, scale=20, size=VECTOR_SIZE))
weights      = np.random.normal(loc=0, scale=1, size=VECTOR_SIZE)
vector_list  = vector.tolist()
weights_list = weights.tolist()

# Precompute plaintext mean and std for normalization constants
# These are treated as known public parameters (not encrypted)
vec_mean = float(np.mean(vector))
vec_std  = float(np.std(vector))

# Avoid division by zero
if vec_std == 0:
    raise ValueError("Standard deviation is zero — cannot normalize.")

inv_std = 1.0 / vec_std

# Warm-up
_ = ts.ckks_vector(context, vector_list)

# ==========================================================
# ENCRYPTION BENCHMARK
# ==========================================================
enc_times = []
for _ in range(REPEATS):
    start   = time.perf_counter()
    enc_vec = ts.ckks_vector(context, vector_list)
    end     = time.perf_counter()
    enc_times.append((end - start) * 1000)

enc_latency_mean = np.mean(enc_times)
enc_latency_std  = np.std(enc_times)

# ==========================================================
# EXECUTION BENCHMARK
# Z-score: (x - mean) * (1/std)  — depth 1 (one scalar mul)
# Dot product: normalized · weights  — depth 1
# Total depth: 2
# Re-encrypt each iteration for consistent state
# ==========================================================
exec_times = []
for _ in range(REPEATS):
    enc_vec    = ts.ckks_vector(context, vector_list)
    start      = time.perf_counter()
    normalized = (enc_vec - vec_mean) * inv_std   # z-score normalization
    dot_result = normalized.dot(weights_list)      # dot product on normalized vector
    end        = time.perf_counter()
    exec_times.append((end - start) * 1000)

exec_latency_mean = np.mean(exec_times)
exec_latency_std  = np.std(exec_times)

# ==========================================================
# DECRYPTION BENCHMARK
# ==========================================================
enc_vec    = ts.ckks_vector(context, vector_list)
normalized = (enc_vec - vec_mean) * inv_std
dot_result = normalized.dot(weights_list)

dec_times = []
for _ in range(REPEATS):
    start      = time.perf_counter()
    dec_result = dot_result.decrypt()
    end        = time.perf_counter()
    dec_times.append((end - start) * 1000)

dec_latency_mean = np.mean(dec_times)
dec_latency_std  = np.std(dec_times)

# ==========================================================
# ACCURACY (MAE vs plaintext)
# ==========================================================
plain_normalized = (vector - vec_mean) * inv_std
plain_dot        = np.dot(plain_normalized, weights)
mae              = abs(dec_result[0] - plain_dot)

# ==========================================================
# CIPHERTEXT SIZE
# ==========================================================
enc_fresh = ts.ckks_vector(context, vector_list)
size_kb   = len(enc_fresh.serialize()) / 1024

# ==========================================================
# OUTPUT
# ==========================================================
print("\n========== PROGRAM 3: HIGH COMPLEXITY ==========")
print(f"Operation                 : z-score normalization + dot product")
print(f"Polynomial Modulus Degree : {POLY_MOD_DEGREE}")
print(f"Vector Size               : {VECTOR_SIZE}")
print(f"Repeats per measurement   : {REPEATS}")
print(f"Vec mean (public)         : {vec_mean:.4f}")
print(f"Vec std  (public)         : {vec_std:.4f}")
print("-------------------------------------------------")
print(f"Encryption latency  (ms)  : {enc_latency_mean:.4f}  ± {enc_latency_std:.4f}")
print(f"Execution latency   (ms)  : {exec_latency_mean:.4f}  ± {exec_latency_std:.4f}")
print(f"Decryption latency  (ms)  : {dec_latency_mean:.4f}  ± {dec_latency_std:.4f}")
print(f"Ciphertext size     (KB)  : {size_kb:.4f}")
print(f"MAE                       : {mae:.10f}")
print("=================================================\n")