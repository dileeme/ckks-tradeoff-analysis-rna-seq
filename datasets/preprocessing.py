import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time

# Install if needed: pip install tqdm --break-system-packages

# ==========================================================
# LOAD
# ==========================================================
print("Loading files...")
with tqdm(total=2, desc="Reading CSVs", ncols=70) as pbar:
    expr = pd.read_csv("datasets/data.csv", index_col=0)
    pbar.update(1)
    labels = pd.read_csv("datasets/labels.csv", index_col=0)
    pbar.update(1)

# ==========================================================
# MERGE
# ==========================================================
with tqdm(total=1, desc="Merging labels", ncols=70) as pbar:
    df = expr.copy()
    df["cancer_type"] = labels["Class"]
    pbar.update(1)

print(f"Merged shape: {df.shape}")
print(f"Cancer type distribution:\n{df['cancer_type'].value_counts()}\n")

# ==========================================================
# FEATURE SELECTION — top 500 by variance
# ==========================================================
gene_cols = [c for c in df.columns if c != "cancer_type"]

print("Computing variance across 20,531 features...")
with tqdm(total=len(gene_cols), desc="Variance calc", ncols=70, unit="gene") as pbar:
    variances = {}
    for col in gene_cols:
        variances[col] = df[col].var()
        pbar.update(1)

top500 = pd.Series(variances).nlargest(500).index.tolist()
df     = df[top500 + ["cancer_type"]]
print(f"After feature selection: {df.shape}\n")

# ==========================================================
# MIN-MAX NORMALIZATION
# ==========================================================
print("Normalizing features...")
with tqdm(total=len(top500), desc="Normalizing", ncols=70, unit="gene") as pbar:
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[top500])
    pbar.update(len(top500))

df[top500] = normalized
print("Normalization complete — values in [0, 1]\n")

# ==========================================================
# BATCH PARTITIONING
# ==========================================================
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

batches = [
    ("datasets/batch_a_100.csv",  df_shuffled.iloc[:100],  "Batch A (100 samples)"),
    ("datasets/batch_b_400.csv",  df_shuffled.iloc[:400],  "Batch B (400 samples)"),
    ("datasets/batch_c_801.csv",  df_shuffled.iloc[:801],  "Batch C (801 samples)"),
    ("datasets/processed_dataset.csv", df_shuffled,        "Full processed dataset"),
]

print("Saving batch files...")
for path, data, label in tqdm(batches, desc="Saving files", ncols=70):
    data.to_csv(path, index=False)

# ==========================================================
# SUMMARY
# ==========================================================
print("\n========== PREPROCESSING COMPLETE ==========")
print(f"Features selected : 500 (top by variance)")
print(f"Normalization     : Min-Max [0, 1]")
print(f"Batch A           : {df_shuffled.iloc[:100].shape}")
print(f"Batch B           : {df_shuffled.iloc[:400].shape}")
print(f"Batch C           : {df_shuffled.iloc[:801].shape}")
print(f"Files saved to    : datasets/")
print("=============================================\n")