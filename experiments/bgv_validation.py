"""
BGV Parameter Validation Script --- Phase 2C
"""
import tenseal as ts
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

SCALE_FACTOR = 10_000
N_FEATURES   = 500
CONFIGS = [
    {"poly_mod_degree": 4096,  "plain_modulus": 786433},
    {"poly_mod_degree": 8192,  "plain_modulus": 786433},
    {"poly_mod_degree": 16384, "plain_modulus": 786433},
]
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_test_vectors():
    np.random.seed(42)
    mean_a = np.random.uniform(0, 1, N_FEATURES)
    mean_b = np.random.uniform(0, 1, N_FEATURES)
    de_int = np.round(np.abs(mean_a - mean_b) * SCALE_FACTOR).astype(np.int64)
    return de_int

def validate_bgv_config(poly_mod_degree, plain_modulus):
    result = {"scheme": "BGV", "poly_mod_degree": poly_mod_degree,
              "plain_modulus": plain_modulus, "slots": poly_mod_degree // 2,
              "slot_capacity_ok": None, "encrypt_ok": None, "depth1_ok": None,
              "roundtrip_exact": None, "max_error": None, "verdict": None, "notes": ""}
    slots = poly_mod_degree // 2
    if slots < N_FEATURES:
        result.update({"slot_capacity_ok": False, "verdict": "EXCLUDED",
                       "notes": f"Only {slots} slots < {N_FEATURES} needed"})
        return result
    result["slot_capacity_ok"] = True
    try:
        ctx = ts.context(ts.SCHEME_TYPE.BGV, poly_modulus_degree=poly_mod_degree, plain_modulus=plain_modulus)
        ctx.generate_galois_keys(); ctx.generate_relin_keys()
    except Exception as e:
        result.update({"encrypt_ok": False, "verdict": "EXCLUDED", "notes": f"Context failed: {e}"})
        return result
    de_int = make_test_vectors()
    padded = de_int.tolist() + [0] * (slots - N_FEATURES)
    try:
        enc = ts.bgv_vector(ctx, padded)
        result["encrypt_ok"] = True
    except Exception as e:
        result.update({"encrypt_ok": False, "verdict": "EXCLUDED", "notes": f"Encrypt failed: {e}"})
        return result
    try:
        dec_scaled = (enc * 2).decrypt()[:N_FEATURES]
        expected = (de_int * 2).tolist()
        ok = all(int(round(d)) == e for d, e in zip(dec_scaled, expected))
        result["depth1_ok"] = ok
        if not ok:
            result.update({"verdict": "EXCLUDED", "notes": "Depth-1 incorrect"})
            return result
    except Exception as e:
        result.update({"depth1_ok": False, "verdict": "EXCLUDED", "notes": f"Depth-1 failed: {e}"})
        return result
    try:
        dec2 = ts.bgv_vector(ctx, padded).decrypt()[:N_FEATURES]
        errors = [abs(int(round(d)) - int(e)) for d, e in zip(dec2, de_int.tolist())]
        result["roundtrip_exact"] = all(e == 0 for e in errors)
        result["max_error"] = max(errors)
        if not result["roundtrip_exact"]:
            result.update({"verdict": "EXCLUDED", "notes": f"Not exact, max err: {max(errors)}"})
            return result
    except Exception as e:
        result.update({"roundtrip_exact": False, "verdict": "EXCLUDED", "notes": f"Roundtrip failed: {e}"})
        return result
    result.update({"verdict": "VALID", "notes": f"All checks passed. Precision: 1/{SCALE_FACTOR}"})
    return result

print("=" * 60)
print("BGV PARAMETER VALIDATION --- Phase 2C")
print("=" * 60)
results = []
for cfg in tqdm(CONFIGS, desc="Validating BGV configs", unit="config"):
    pmd, pm = cfg["poly_mod_degree"], cfg["plain_modulus"]
    tqdm.write(f"\nTesting poly_mod_degree={pmd}...")
    r = validate_bgv_config(pmd, pm)
    results.append(r)
    tqdm.write(f"  Slots: {r['slots']} | Verdict: {r['verdict']} | {r['notes']}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "bgv_validation_results.csv"), index=False)
print("\n" + "=" * 60)
print(df[["scheme", "poly_mod_degree", "plain_modulus", "slots", "verdict"]].to_string(index=False))
