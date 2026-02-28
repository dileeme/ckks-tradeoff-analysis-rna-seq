

# 🔐 CKKS / BFV / BGV FHE Tradeoff Analysis
### Differential Expression Scoring on RNA-Seq Gene Expression Data

> **Author:** Dilen Shankar  
> **Target:** IEEE T-IFS / Journal of Biomedical Informatics / BMC Bioinformatics  
> **Stack:** TenSEAL · Microsoft SEAL · scikit-learn · pandas · NumPy · Matplotlib

---

## 📋 Overview

This project benchmarks three Fully Homomorphic Encryption (FHE) schemes — **CKKS**, **BFV**, and **BGV** — for computing differential expression (DE) scores on RNA-Seq cancer gene expression data under encryption. The goal is to produce a systematic tradeoff analysis of computational performance, ciphertext overhead, and approximation accuracy across varying polynomial modulus degrees and dataset sizes.

The encrypted operation is a **depth-1 computation**: encrypted group mean per cancer type followed by ciphertext subtraction to produce DE scores — without ever decrypting the data mid-computation.

---

## 🧬 Research Question

> How do FHE scheme choice (CKKS, BFV, BGV), polynomial modulus degree, and dataset size influence computational performance, noise/overflow behavior, ciphertext overhead, and approximation accuracy when applied to differential expression scoring on RNA-Seq gene expression data?

---

## 📊 Datasets

### Dataset 1 — UCI Gene Expression Cancer RNA-Seq
| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (ID 401) |
| Citation | Fiorini (2016); Weinstein et al. *Nature Genetics* (2013) |
| Shape | 801 samples × 20,531 features |
| Cancer Types | BRCA (300), KIRC (146), LUAD (141), PRAD (136), COAD (78) |
| Features | Anonymous identifiers — gene_0 through gene_20530 |
| Missing Values | None |
| Status | ✅ Loaded, preprocessed, batched, baselines computed |

### Dataset 2 — NCBI GEO RNA-Seq
| Property | Value |
|---|---|
| Source | NCBI Gene Expression Omnibus (GEO) |
| Requirements | RNA-Seq cancer expression matrix, multiple tumor types, HGNC gene identifiers preferred |
| Purpose | Eliminates single-dataset generalizability objection for journal reviewers |
| Status | ⏳ Not yet sourced — Day 1 task |

---

## ⚙️ Preprocessing Pipeline (Dataset 1 — Complete)

```
data.csv (801×20531) ──┐
                        ├──► merge on sample index ──► top-500 variance features
labels.csv (801×1)  ──┘         ──► min-max normalize [0,1] ──► shuffle (seed=42)
                                        ──► batch_a (100) / batch_b (400) / batch_c (801)
```

> **Why min-max over log2?** Paper is 60% systems focused. Min-max guarantees CKKS numerical stability. Log2 produces unbounded negative values at low expression levels which destabilize CKKS encoding.

---

## 🔬 Encryption Schemes

| Scheme | Arithmetic | MAE Expected | Notes |
|---|---|---|---|
| CKKS | Approximate float | Small (< 0.12 at 8192+) | Primary scheme — deep characterization |
| BFV | Exact integer | 0 (or total failure) | DE scores scaled to int before encryption |
| BGV | Exact integer | 0 (or total failure) | Same scaling approach as BFV |

> **BFV/BGV note:** Float DE scores are multiplied by 10⁴ or 10⁶, rounded, encrypted, decrypted, then rescaled. This must be documented explicitly in methodology as it affects cross-scheme MAE comparisons.

---

## 📐 Computation Graph

```
Inputs:  Encrypted sample vectors (500 features each)
         Plaintext group membership labels
         Plaintext scalar 1/n per group

Step 1:  Σ enc(xᵢ[f])  for i in group_A  →  encrypted sum_A       [depth: 0]
Step 2:  sum_A × (1/n_A)                  →  encrypted mean_A      [depth: 1]
Step 3:  mean_A - mean_B                  →  encrypted DE score(f)  [depth: 0]

Total multiplicative depth: 1
Compatible moduli: 8192, 16384
```

---

## 🧪 Experiment Matrix

**Total: 36 configurations × 10 runs = 360 runs**

| Configs | Scheme | Dataset | Moduli | Batches | Runs each |
|---|---|---|---|---|---|
| 1–6   | CKKS | Dataset 1 | 8192, 16384 | 100, 400, 801 | 10 |
| 7–12  | CKKS | Dataset 2 | 8192, 16384 | TBD | 10 |
| 13–18 | BFV  | Dataset 1 | 8192, 16384 | 100, 400, 801 | 10 |
| 19–24 | BFV  | Dataset 2 | 8192, 16384 | TBD | 10 |
| 25–30 | BGV  | Dataset 1 | 8192, 16384 | 100, 400, 801 | 10 |
| 31–36 | BGV  | Dataset 2 | 8192, 16384 | TBD | 10 |

> **Note:** poly_mod_degree=4096 is excluded for CKKS — catastrophic MAE at scale=2³⁰ (formally documented in Phase 1). BFV/BGV 4096 exclusion TBD from parameter validation.

---

## 📏 Metrics Per Run

| Metric | Description |
|---|---|
| `enc_latency_ms` | Time to encrypt one batch of sample vectors |
| `exec_latency_ms` | Time to compute encrypted DE scores |
| `dec_latency_ms` | Time to decrypt results |
| `ct_size_kb` | Ciphertext size on disk |
| `mae` | Mean absolute error vs plaintext baseline (CKKS only — BFV/BGV should be 0) |

---

## 📁 Project Structure

```
ckks-tradeoff-analysis-rna-seq/
│
├── datasets/
│   ├── data.csv                        # Raw expression matrix (801×20531)
│   ├── labels.csv                      # Cancer type labels (801×1)
│   ├── batch_a_100.csv                 # 100 samples, normalized, top-500 features
│   ├── batch_b_400.csv                 # 400 samples
│   ├── batch_c_801.csv                 # 801 samples (full)
│   ├── processed_dataset.csv           # Master shuffled reference file
│   └── de_baselines/
│       ├── de_baseline_batch_a.csv     # Plaintext DE scores — batch A (500×11)
│       ├── de_baseline_batch_b.csv     # Plaintext DE scores — batch B (500×11)
│       └── de_baseline_batch_c.csv     # Plaintext DE scores — batch C (500×11)
│
├── experiments/
│   └── phase3_ckks_de.py               # Phase 3 CKKS experiment ✅ READY TO RUN
│
└── results/
    └── phase3_results.csv              # Phase 3 output (created by script)
```

---

## 🚦 Phase Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Pilot study — synthetic CKKS benchmarks, primary hypothesis | ✅ Complete |
| Phase 2 | Dataset 1 prep, batching, plaintext DE baseline | ✅ Complete |
| Phase 2B | Dataset 2 prep, batching, plaintext DE baseline | ⏳ Not started |
| Phase 2C | BFV/BGV parameter validation | ⏳ Not started |
| Phase 3 | CKKS experimentation — 6 configs × 10 runs — Dataset 1 | 🔜 Next |
| Phase 3B | Full matrix — CKKS + BFV + BGV — both datasets | ⏳ Pending |
| Phase 4 | Analysis, predictive model, visualizations | ⏳ Pending |
| Phase 5 | Literature survey | ⏳ Pending — start during Phase 3 compute time |

---

## 🏃 Running Phase 3

```bash
# Activate environment
cd ckks-tradeoff-analysis-rna-seq
.\ckks_env\Scripts\activate        # Windows
source ckks_env/bin/activate       # Linux/Mac

# Run Phase 3 CKKS experiments
python experiments/phase3_ckks_de.py
```

> Results are written to `results/phase3_results.csv` **after every single run** — no data loss if it crashes mid-way. Config 6 (16384 × 801 samples) will be the slowest — run it overnight.

---

## 📚 Key References

| Paper | Relevance |
|---|---|
| Weinstein et al. *Nature Genetics* (2013) — TCGA PANCAN | Primary dataset source — 6000+ citations |
| Fiorini (2016) — UCI ML Repository | Dataset 1 curation |
| Blatt et al. *Medical Genomics* (2020) — CKKS for GWAS | Closest prior work — cite and differentiate |
| Sim et al. *Medical Genomics* (2020) — GWAS with HE | Background — cite |
| Namazi et al. (2025) — Multi-key HE for genomics | Recent overlap — read carefully |
| Abinaya & Santhi (2021) — Survey on genomic privacy | Background survey — cite |

---

## 🎯 Publication Target

| Journal | Tier | Notes |
|---|---|---|
| IEEE Transactions on Information Forensics & Security | Top-tier stretch | Strong FHE readership |
| Journal of Biomedical Informatics | Mid-tier strong | Best fit with two datasets |
| BMC Bioinformatics | Mid-tier achievable | Good impact factor |
| Computers & Security | Safe fallback | Likely acceptance with full scope |

**Estimated journal publication score (full scope):** `87 / 100`

---

*Last updated: Phase 2 complete. Phase 3 ready to run.*
