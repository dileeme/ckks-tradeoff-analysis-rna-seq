"""
topk_analysis.py  —  compute top-k precision from saved score arrays

Run after both topk_bfv_de.py and topk_ckks_de.py have completed.

Outputs:
  results/topk/topk_precision_results.csv   — full results table
  Prints the LaTeX table for Section 3.6 directly to console
"""

import numpy as np
import pandas as pd
import os

K_VALUES = [50, 100, 200, 500]

CONFIGS = [
    {"scheme": "BFV",  "dataset": "D1", "prefix": "results/topk/bfv_d1"},
    {"scheme": "BFV",  "dataset": "D2", "prefix": "results/topk/bfv_d2"},
    {"scheme": "CKKS", "dataset": "D1", "prefix": "results/topk/ckks_d1"},
    {"scheme": "CKKS", "dataset": "D2", "prefix": "results/topk/ckks_d2"},
]

rows = []

for cfg in CONFIGS:
    prefix = cfg["prefix"]
    plain_path = f"{prefix}_plain_scores.npy"
    enc_path   = f"{prefix}_enc_scores_10runs.npy"

    if not os.path.exists(plain_path) or not os.path.exists(enc_path):
        print(f"MISSING: {prefix} — skipping")
        continue

    plain = np.load(plain_path)   # (n_genes, n_pairs)
    enc   = np.load(enc_path)     # (n_runs, n_genes, n_pairs)

    n_genes, n_pairs = plain.shape
    n_runs = enc.shape[0]

    print(f"\n{cfg['scheme']} {cfg['dataset']}: "
          f"{n_genes} genes, {n_pairs} pairs, {n_runs} runs")

    for k in K_VALUES:
        if k > n_genes:
            continue

        # Top-k plain genes: aggregate across pairs by mean score
        plain_mean = plain.mean(axis=1)               # (n_genes,)
        top_k_plain = set(np.argsort(plain_mean)[-k:])

        # Per run: get top-k encrypted genes, compute precision@k
        precisions = []
        for run in range(n_runs):
            enc_mean = enc[run].mean(axis=1)          # (n_genes,)
            top_k_enc = set(np.argsort(enc_mean)[-k:])
            precision = len(top_k_plain & top_k_enc) / k
            precisions.append(precision)

        mean_prec = np.mean(precisions)
        std_prec  = np.std(precisions)

        print(f"  k={k:4d}: precision@k = {mean_prec:.4f} ± {std_prec:.4f}  "
              f"({mean_prec*100:.1f}%)")

        rows.append({
            "scheme":      cfg["scheme"],
            "dataset":     cfg["dataset"],
            "k":           k,
            "precision_mean": round(mean_prec, 4),
            "precision_std":  round(std_prec,  4),
            "overlap_count":  round(mean_prec * k, 1),
        })

df = pd.DataFrame(rows)
os.makedirs("results/topk", exist_ok=True)
df.to_csv("results/topk/topk_precision_results.csv", index=False)
print(f"\nSaved: results/topk/topk_precision_results.csv")

# ── LaTeX table for paper ──────────────────────────────────────────────────
print("\n" + "="*65)
print("LATEX TABLE — paste into Section 3.6 of sn-article.tex")
print("="*65)
print()
print(r"\begin{table}[!htbp]")
print(r"\small")
print(r"\caption{Top-$k$ precision of encrypted versus plaintext DE gene")
print(r"rankings for BFV and CKKS at $N=8192$, batch\_c configurations,")
print(r"averaged across 10 independent runs ($\pm 1$ standard deviation).}")
print(r"\label{tab:topk}")
print(r"\begin{tabular}{llrrrr}")
print(r"\hline")
print(r"\textbf{Scheme} & \textbf{Dataset} & "
      r"\textbf{$k=50$} & \textbf{$k=100$} & "
      r"\textbf{$k=200$} & \textbf{$k=500$} \\")
print(r"\hline")

for scheme in ["BFV", "CKKS"]:
    for ds in ["D1", "D2"]:
        sub = df[(df["scheme"] == scheme) & (df["dataset"] == ds)]
        if sub.empty:
            continue
        cells = []
        for k in K_VALUES:
            row = sub[sub["k"] == k]
            if row.empty:
                cells.append("---")
            else:
                m = row["precision_mean"].values[0]
                s = row["precision_std"].values[0]
                if s < 0.0005:
                    cells.append(f"{m:.4f}")
                else:
                    cells.append(f"${m:.4f} \\pm {s:.4f}$")
        ds_label = "Dataset~1" if ds == "D1" else "Dataset~2"
        print(f"{scheme} & {ds_label} & " + " & ".join(cells) + r" \\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

# ── Console summary for paper paragraph ───────────────────────────────────
print("\n" + "="*65)
print("SUMMARY FOR PAPER PARAGRAPH (Section 3.6)")
print("="*65)
for scheme in ["BFV", "CKKS"]:
    for ds in ["D1", "D2"]:
        sub = df[(df["scheme"] == scheme) & (df["dataset"] == ds)]
        if sub.empty:
            continue
        print(f"\n{scheme} {ds}:")
        for _, r in sub.iterrows():
            print(f"  top-{int(r['k'])}: {r['precision_mean']*100:.1f}% "
                  f"(±{r['precision_std']*100:.1f}%)")