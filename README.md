# 🔐 BFV / CKKS FHE Tradeoff Analysis
### Differential Expression Scoring on RNA-Seq Gene Expression Data

> **Author:** Dilen Shankar  
> **Target:** BMC Bioinformatics (Research Article — Transcriptome analysis)  
> **Stack:** TenSEAL · Microsoft SEAL · scikit-learn · pandas · NumPy · Matplotlib

---

## 📋 Overview

This project benchmarks two Fully Homomorphic Encryption (FHE) schemes — **BFV** and **CKKS** — for computing differential expression (DE) scores on RNA-Seq cancer gene expression data under encryption. The goal is to produce a systematic tradeoff analysis of computational performance, ciphertext storage overhead, and approximation accuracy across varying polynomial modulus degrees and dataset sizes.

The encrypted operation is a **depth-1 computation**: encrypted group mean per cancer type followed by ciphertext subtraction to produce DE scores — without ever decrypting the data mid-computation.

---

## 🧬 Research Question

> How do FHE scheme choice (BFV vs CKKS), polynomial modulus degree, and dataset size influence computational performance, ciphertext storage overhead, and approximation accuracy when applied to differential expression scoring on RNA-Seq gene expression data?

---

## 📊 Datasets

### Dataset 1 — UCI Gene Expression Cancer RNA-Seq
| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (ID 401) |
| Shape | 801 samples × 20,531 features |
| Cancer Types | BRCA, KIRC, COAD, LUAD, PRAD |
| Pairwise Comparisons | C(5,2) = 10 |
| Cohort Sizes | n ∈ {100, 400, 801} |
| Status | ✅ Complete |

### Dataset 2 — TCGA LUSC+LUAD
| Property | Value |
|---|---|
| Source | The Cancer Genome Atlas (TCGA), accessed via UCSC Xena Browser |
| Shape | 1,129 samples × 20,531 features |
| Cancer Types | LUSC, LUAD |
| Pairwise Comparisons | C(2,2) = 1 |
| Cohort Sizes | n ∈ {100, 400, 1129} |
| Status | ✅ Complete |

---

## ⚙️ Preprocessing Pipeline

```
Raw expression matrix
    ──► log2(x + 1) transform          # variance stabilisation
    ──► zero-variance feature removal   # plaintext only
    ──► z-score normalisation           # per sample, zero mean / unit variance
    ──► batch at n ∈ {100, 400, max}   # three cohort sizes per dataset
```

All preprocessing performed in plaintext prior to encryption. No preprocessing operations performed under encryption.

---

## 🔬 Encryption Schemes

| Scheme | Arithmetic | PMD Tested | Notes |
|---|---|---|---|
| CKKS | Approximate float | 8192, 16384 | N=4096 excluded — insufficient modulus headroom at 128-bit security |
| BFV | Exact integer | 4096, 8192, 16384 | Float DE scores scaled by fixed integer factor prior to encoding |

All configurations verified compliant with **128-bit classical security** as defined by the HE Standard (Albrecht et al., 2019) and enforced by Microsoft SEAL parameter validation.

---

## 📐 HE Parameter Configurations

| Scheme | N (PMD) | Coefficient Modulus | Scale | Security |
|---|---|---|---|---|
| CKKS | 8,192 | [40,30,30,40] = 140 bits | 2^30 | 128-bit |
| CKKS | 16,384 | [60,40,40,60] = 200 bits | 2^40 | 128-bit |
| BFV | 4,096 | SEAL default ≤ 109 bits | N/A | 128-bit |
| BFV | 8,192 | SEAL default ≤ 218 bits | N/A | 128-bit |
| BFV | 16,384 | SEAL default ≤ 438 bits | N/A | 128-bit |

---

## 🧪 Experiment Matrix

**Total: 300 runs (5 configs × 3 cohort sizes × 2 datasets × 10 runs)**

| Configs | Scheme | Dataset | PMD | Batches | Runs |
|---|---|---|---|---|---|
| 1–6   | CKKS | Dataset 1 | 8192, 16384 | 100, 400, 801 | 10 |
| 7–12  | CKKS | Dataset 2 | 8192, 16384 | 100, 400, 1129 | 10 |
| 13–18 | BFV  | Dataset 1 | 4096, 8192, 16384 | 100, 400, 801 | 10 |
| 19–24 | BFV  | Dataset 2 | 4096, 8192, 16384 | 100, 400, 1129 | 10 |

---

## 📏 Metrics Per Run

| Metric | Description |
|---|---|
| `enc_latency_ms` | Wall-clock time to encode and encrypt all samples in cohort |
| `exec_latency_ms` | Wall-clock time to compute all homomorphic DE scoring operations |
| `dec_latency_ms` | Wall-clock time to decrypt and decode all results |
| `ct_size_kb` | Serialised size of a single encrypted sample vector (KB) |
| `mae` | Mean absolute error vs plaintext baseline, per gene across all pairwise comparisons |

All metrics averaged across 10 runs and reported with ±1 standard deviation. Total latency = enc + exec + dec.

---

## 📁 Project Structure


project-root/
│
├── datasets/
│   ├── batch_a_100.csv
│   ├── batch_b_400.csv
│   ├── batch_c_801.csv
│   ├── dataset2_batch_a_100.csv
│   ├── dataset2_batch_b_400.csv
│   ├── dataset2_batch_c_1129.csv
│   ├── data.csv
│   ├── labels.csv
│   └── de_baselines/
│       ├── de_baseline_batch_a.csv
│       ├── de_baseline_batch_b.csv
│       ├── de_baseline_batch_c.csv
│       ├── de_baseline_dataset2_batch_a.csv
│       ├── de_baseline_dataset2_batch_b.csv
│       └── de_baseline_dataset2_batch_c.csv
│
├── experiments/
│   ├── phase3_ckks_de.py
│   └── phase3_bfv_de.py
│
├── figures/
│   ├── plot1_enc_latency.png
│   ├── plot2_exec_latency.png
│   ├── plot3_ct_size.png
│   ├── plot4_total_latency.png
│   └── plot5_mae.png
│
├── results/
│   ├── phase3_ckks_dataset1.csv
│   ├── phase3_ckks_dataset2.csv
│   ├── phase3_bfv_dataset1.csv
│   └── phase3_bfv_dataset2.csv
│
├── manuscript/
│   ├── sn-article.tex
│   ├── sn-article.pdf
│   └── sn-bibliography.bib
│
└── README.md



## 🚦 Phase Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Pilot study — CKKS parameter validation, 4096 exclusion documented | ✅ Complete |
| Phase 2 | Dataset 1 prep, batching, plaintext DE baselines | ✅ Complete |
| Phase 2B | Dataset 2 (TCGA LUSC+LUAD via Xena) prep and baselines | ✅ Complete |
| Phase 3 | CKKS experiments — both datasets | ✅ Complete |
| Phase 3B | BFV experiments — both datasets | ✅ Complete |
| Phase 4 | Analysis, visualisations (5 plots) | ✅ Complete |
| Phase 5 | Literature review, novelty positioning | ✅ Complete |
| Manuscript | Methods, Results, Abstract drafted; Discussion/Conclusion pending | 🔜 In progress |

--

## 📊 Key Results Summary

| Scheme | N | Dataset | Total Latency (ms) | CT Size (KB) | MAE |
|---|---|---|---|---|---|
| BFV | 4,096 | D1 | 577.80 | 128 | 2–7 × 10⁻⁶ |
| BFV | 8,192 | D1 | 1,680.89 | 512 | 2–7 × 10⁻⁶ |
| BFV | 16,384 | D1 | 5,211.36 | 2,048 | 2–7 × 10⁻⁶ |
| CKKS | 8,192 | D1 | 8,788.41 | 244 | 1.2–1.3 × 10⁻⁵ |
| CKKS | 16,384 | D1 | 38,453.92 | 770 | 2.5 × 10⁻⁵ |

**Key findings:**
- BFV achieves **3.5–7.5× lower total latency** than CKKS across all configurations
- CKKS ciphertexts are **2.66× smaller** per sample at N=16384 — clear latency–storage tradeoff
- Execution cost scales with **number of pairwise comparisons C(K,2)**, not sample count
- CKKS MAE **inverts** with increasing N — larger scale (2^40) introduces more rescaling noise for depth-1 computation
- BFV MAE is **~10× lower** than CKKS and decreases with cohort size via quantisation noise averaging

---

## 🖥️ Environment

| Property | Value |
|---|---|
| CPU | Intel Core i7-10750H @ 2.60 GHz (6 cores, 12 threads) |
| RAM | 16 GB |
| OS | Windows 11, PowerShell via VS Code |
| Python | 3.12 |
| FHE Library | TenSEAL 0.3.16 (Microsoft SEAL backend) |
| Threading | OpenMP multi-core, thread count not pinned |

---

## 📚 Key References

| Reference | Role |
|---|---|
| Albrecht et al. (2019) — HE Standard | Security parameter compliance |
| Benaissa et al. (2021) — TenSEAL | FHE library citation |
| Microsoft SEAL | Backend library citation |
| Goldman et al. (2020) — UCSC Xena Browser | Dataset 2 access portal |
| Gürsoy et al. (2025) — pQuant, Nature Comms | HE for RNA-seq quantification (differentiate: they do quantification, we do DE benchmarking) |
| Namazi et al. (2025) — Multi-key HE, Bioinformatics Oxford | Recent overlap — cited and differentiated |
| Jiang et al. (2022) — FHEBench | Primary benchmarking gap anchor |
| Krüger et al. (2025) — CKKS vs TFHE | Most recent scheme comparison |

---

## 🎯 Publication Target

**BMC Bioinformatics** — Research Article, Transcriptome analysis category  
IF: 3.3 (2024) | Desk decision median: ~5 days | Full pipeline: ~4–6 months

---

*Last updated: Manuscript in progress. Methods and Results sections complete. Discussion and Conclusion pending.*
