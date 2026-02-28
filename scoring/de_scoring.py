import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
import os

# ==========================================================
# CONFIGURATION
# ==========================================================
BATCHES = {
    "batch_a": "datasets/batch_a_100.csv",
    "batch_b": "datasets/batch_b_400.csv",
    "batch_c": "datasets/batch_c_801.csv",
}
OUTPUT_DIR = "scoring/de_baselines/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_TYPES = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]

# ==========================================================
# DE SCORING FUNCTION
# ==========================================================
def compute_de_scores(df, batch_name):
    """
    For each pair of cancer types, compute mean expression
    difference per gene (plaintext DE score).
    """
    gene_cols = [c for c in df.columns if c != "cancer_type"]
    pairs     = list(combinations(CANCER_TYPES, 2))
    results   = {"gene": gene_cols}

    print(f"\nComputing DE scores for {batch_name}...")
    print(f"  Samples : {len(df)}")
    print(f"  Genes   : {len(gene_cols)}")
    print(f"  Pairs   : {len(pairs)}")

    for type_a, type_b in tqdm(pairs, desc="  Cancer pairs", ncols=70):
        group_a = df[df["cancer_type"] == type_a][gene_cols]
        group_b = df[df["cancer_type"] == type_b][gene_cols]

        # Skip pair if either group is empty in this batch
        if len(group_a) == 0 or len(group_b) == 0:
            print(f"  WARNING: {type_a} or {type_b} not present in {batch_name} — skipping pair")
            continue

        mean_a = group_a.mean()
        mean_b = group_b.mean()

        de_score = (mean_a - mean_b).abs()
        results[f"{type_a}_vs_{type_b}"] = de_score.values

    return pd.DataFrame(results)

# ==========================================================
# RUN FOR EACH BATCH
# ==========================================================
summary = {}

for batch_name, batch_path in BATCHES.items():
    print(f"\n{'='*50}")
    print(f"Processing {batch_name}...")

    df = pd.read_csv(batch_path)

    # Show class distribution for this batch
    dist = df["cancer_type"].value_counts()
    print(f"  Class distribution:\n{dist.to_string()}")

    # Compute DE scores
    de_scores = compute_de_scores(df, batch_name)

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"de_baseline_{batch_name}.csv")
    de_scores.to_csv(out_path, index=False)

    # Summary stats
    score_cols = [c for c in de_scores.columns if c != "gene"]
    mean_de    = de_scores[score_cols].values.mean()
    max_de     = de_scores[score_cols].values.max()
    min_de     = de_scores[score_cols].values.min()

    summary[batch_name] = {
        "samples"     : len(df),
        "pairs"       : len(score_cols),
        "mean_DE"     : round(mean_de, 6),
        "max_DE"      : round(max_de, 6),
        "min_DE"      : round(min_de, 6),
        "output_file" : out_path,
    }

    print(f"  Mean DE score : {mean_de:.6f}")
    print(f"  Max DE score  : {max_de:.6f}")
    print(f"  Min DE score  : {min_de:.6f}")
    print(f"  Saved to      : {out_path}")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print(f"\n{'='*50}")
print("PLAINTEXT DE BASELINE COMPLETE")
print(f"{'='*50}")
for batch, stats in summary.items():
    print(f"\n{batch}:")
    for k, v in stats.items():
        print(f"  {k:15}: {v}")