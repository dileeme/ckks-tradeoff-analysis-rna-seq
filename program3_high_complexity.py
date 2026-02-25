import numpy as np
import time
import os
import tenseal as ts
from ckks_utils import create_context

# ==========================================================
# CONFIGURATION
# ==========================================================

POLY_MOD_DEGREE = 8192     # Test: 8192 first, then 16384
VECTOR_SIZE = 4096
REPEATS = 10               # Heavy workload

np.random.seed(42)

# ==========================================================
# CONTEXT SETUP
# ==========================================================

context = create_context(POLY_MOD_DEGREE)

# ==========================================================
# SYNTHETIC RNA-LIKE DATA
# ==========================================================

vector = np.abs(
    np.random.normal(loc=50, scale=20, size=VECTOR_SIZE)
)

# Z-score normalization (plaintext)
mean = np.mean(vector)
std = np.std(vector)
vector = (vector - mean) / std

# ==========================================================
# ENCRYPTION BENCHMARK
# ==========================================================

start_enc = time.perf_counter()

for _ in range(REPEATS):
    enc_vec = ts.ckks_vector(context, vector.tolist())

end_enc = time.perf_counter()
enc_latency = (end_enc - start_enc) * 1000 / REPEATS

enc_vec = ts.ckks_vector(context, vector.tolist())

# ==========================================================
# EXECUTION BENCHMARK
#   1) Aggregate via sum
#   2) Degree-5 sigmoid approximation
# ==========================================================

start_exec = time.perf_counter()

for _ in range(REPEATS):

    x = enc_vec.sum()

    x3 = x * x * x
    x5 = x3 * x * x

    # Chebyshev-style sigmoid approximation
    poly = 0.5 + 0.215919 * x - 0.008217 * x3 + 0.000182 * x5

end_exec = time.perf_counter()
exec_latency = (end_exec - start_exec) * 1000 / REPEATS

# ==========================================================
# DECRYPTION BENCHMARK
# ==========================================================

x = enc_vec.sum()
x3 = x * x * x
x5 = x3 * x * x
poly = 0.5 + 0.215919 * x - 0.008217 * x3 + 0.000182 * x5

start_dec = time.perf_counter()

for _ in range(REPEATS):
    dec_result = poly.decrypt()

end_dec = time.perf_counter()
dec_latency = (end_dec - start_dec) * 1000 / REPEATS

# ==========================================================
# PLAINTEXT REFERENCE
# ==========================================================

plain_x = np.sum(vector)
plain_poly = (
    0.5
    + 0.215919 * plain_x
    - 0.008217 * (plain_x ** 3)
    + 0.000182 * (plain_x ** 5)
)

mae = abs(dec_result[0] - plain_poly)

# ==========================================================
# CIPHERTEXT SIZE
# ==========================================================

serialized = enc_vec.serialize()

with open("temp_cipher3.bin", "wb") as f:
    f.write(serialized)

size_kb = os.path.getsize("temp_cipher3.bin") / 1024
os.remove("temp_cipher3.bin")

# ==========================================================
# OUTPUT
# ==========================================================

print("\n========== PROGRAM 3: HIGH COMPLEXITY ==========")
print(f"Polynomial Modulus Degree: {POLY_MOD_DEGREE}")
print(f"Vector Size: {VECTOR_SIZE}")
print(f"Repeats per measurement: {REPEATS}")
print("--------------------------------------------------")
print(f"Encryption latency (ms): {enc_latency:.4f}")
print(f"Execution latency (ms): {exec_latency:.4f}")
print(f"Decryption latency (ms): {dec_latency:.4f}")
print(f"Ciphertext size (KB): {size_kb:.4f}")
print(f"MAE: {mae:.10f}")
print("==================================================\n")