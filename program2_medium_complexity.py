import numpy as np
import time
import os
import tenseal as ts
from ckks_utils import create_context

# ==========================================================
# CONFIGURATION
# ==========================================================

POLY_MOD_DEGREE = 8192   # Test: 4096, 8192, 16384
VECTOR_SIZE = 512
REPEATS = 20             # Fewer repeats (heavier workload)

np.random.seed(42)

# ==========================================================
# CONTEXT SETUP
# ==========================================================

context = create_context(POLY_MOD_DEGREE)

# ==========================================================
# SYNTHETIC DATA
# ==========================================================

vector = np.abs(
    np.random.normal(loc=50, scale=20, size=VECTOR_SIZE)
)

weights = np.random.normal(
    loc=0,
    scale=1,
    size=VECTOR_SIZE
)

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
#   1) Encrypted dot product
#   2) Degree-3 polynomial
# ==========================================================

start_exec = time.perf_counter()

for _ in range(REPEATS):

    dot_product = enc_vec.dot(weights.tolist())

    # f(x) = 0.5 + 0.197x − 0.004x^3
    x = dot_product
    x3 = x * x * x
    poly = 0.5 + 0.197 * x - 0.004 * x3

end_exec = time.perf_counter()
exec_latency = (end_exec - start_exec) * 1000 / REPEATS

# ==========================================================
# DECRYPTION BENCHMARK
# ==========================================================

dot_product = enc_vec.dot(weights.tolist())
x = dot_product
x3 = x * x * x
poly = 0.5 + 0.197 * x - 0.004 * x3

start_dec = time.perf_counter()

for _ in range(REPEATS):
    dec_result = poly.decrypt()

end_dec = time.perf_counter()
dec_latency = (end_dec - start_dec) * 1000 / REPEATS

# ==========================================================
# PLAINTEXT REFERENCE
# ==========================================================

plain_dot = np.dot(vector, weights)
plain_poly = 0.5 + 0.197 * plain_dot - 0.004 * (plain_dot ** 3)

mae = abs(dec_result[0] - plain_poly)

# ==========================================================
# CIPHERTEXT SIZE
# ==========================================================

serialized = enc_vec.serialize()

with open("temp_cipher2.bin", "wb") as f:
    f.write(serialized)

size_kb = os.path.getsize("temp_cipher2.bin") / 1024
os.remove("temp_cipher2.bin")

# ==========================================================
# OUTPUT
# ==========================================================

print("\n========== PROGRAM 2: MEDIUM COMPLEXITY ==========")
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