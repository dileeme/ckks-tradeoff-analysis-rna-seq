import numpy as np
import time
import os
import tenseal as ts
from ckks_utils import create_context

# ==========================================================
# CONFIGURATION
# ==========================================================

POLY_MOD_DEGREE = 16384  # Change to 4096 or 16384
VECTOR_SIZE = 64
REPEATS = 50             # Amplify timing signal

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


# ==========================================================
# ENCRYPTION BENCHMARK
# ==========================================================

start_enc = time.perf_counter()

for _ in range(REPEATS):
    enc_vec = ts.ckks_vector(context, vector.tolist())

end_enc = time.perf_counter()
enc_latency = (end_enc - start_enc) * 1000 / REPEATS

# Use one encrypted vector for execution tests
enc_vec = ts.ckks_vector(context, vector.tolist())

# ==========================================================
# EXECUTION BENCHMARK (Add + Scalar Multiply)
# ==========================================================

start_exec = time.perf_counter()

for _ in range(REPEATS):
    result = enc_vec + enc_vec
    result = result * 0.5

end_exec = time.perf_counter()
exec_latency = (end_exec - start_exec) * 1000 / REPEATS

# ==========================================================
# DECRYPTION BENCHMARK
# ==========================================================

result = enc_vec + enc_vec
result = result * 0.5

start_dec = time.perf_counter()

for _ in range(REPEATS):
    dec_result = result.decrypt()

end_dec = time.perf_counter()
dec_latency = (end_dec - start_dec) * 1000 / REPEATS

# ==========================================================
# ACCURACY (MAE)
# ==========================================================

plain_result = (vector + vector) * 0.5
mae = np.mean(np.abs(np.array(dec_result) - plain_result))

# ==========================================================
# CIPHERTEXT SIZE
# ==========================================================

serialized = enc_vec.serialize()

with open("temp_cipher.bin", "wb") as f:
    f.write(serialized)

size_kb = os.path.getsize("temp_cipher.bin") / 1024
os.remove("temp_cipher.bin")

# ==========================================================
# OUTPUT
# ==========================================================

print("\n========== PROGRAM 1: LOW COMPLEXITY ==========")
print(f"Polynomial Modulus Degree: {POLY_MOD_DEGREE}")
print(f"Vector Size: {VECTOR_SIZE}")
print(f"Repeats per measurement: {REPEATS}")
print("-----------------------------------------------")
print(f"Encryption latency (ms): {enc_latency:.4f}")
print(f"Execution latency (ms): {exec_latency:.4f}")
print(f"Decryption latency (ms): {dec_latency:.4f}")
print(f"Ciphertext size (KB): {size_kb:.4f}")
print(f"MAE: {mae:.10f}")
print("===============================================\n")