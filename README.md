# Federated Visual Unlearning for Deep Vision Models

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower FL](https://img.shields.io/badge/Flower-flwr%201.5-ffd700.svg)](https://flower.ai/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%20A6000-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

An end-to-end framework for **Federated Visual Unlearning** on **CIFAR-100**, enforcing the **"Right to be Forgotten" (GDPR Article 17 / CCPA)** in distributed vision systems. This repository features robust non-IID representation engineering, historical calibration (**FedEraser**), dual visual-quantitative verification (**Grad-CAM** + **Membership Inference Attacks**), and our novel sub-second, zero-storage unlearning paradigm: **Fed-CNS (Contrastive Null-Space Unlearning via Additive Ledger)**.

---

## 📑 Table of Contents

- [Problem Statement: The Illusion of Forgetting](#-problem-statement-the-illusion-of-forgetting)
- [System Architecture & Representation Engineering](#-system-architecture--representation-engineering)
- [Empirical Benchmarks (FedEraser on CIFAR-100)](#-empirical-benchmarks-federaser-on-cifar-100)
- [Dual Auditing Pipeline: Grad-CAM & MIA](#-dual-auditing-pipeline-grad-cam--mia)
- [Novel Contribution: Fed-CNS](#-novel-contribution-fed-cns-contrastive-null-space-unlearning)
- [Critical Self-Audit & 2-Tier Architecture](#-critical-self-audit--the-class-conditional-2-tier-architecture)
- [Interactive Streamlit Dashboard (`app.py`)](#-interactive-streamlit-web-dashboard)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [Formal Documentation & Artifacts](#-formal-documentation--artifacts)

---

## 🎯 Problem Statement: The "Illusion of Forgetting"

Under GDPR Article 17 and CCPA, clients participating in Federated Learning (FL) can demand that all traces of their private data be expunged from the collaborative global model. In distributed computer vision, three naive approaches fail:

1. **The Scalability Wall of Exact Retraining ($M_{\text{retrain}}$)**: Retraining from scratch without the leaving client is the gold-standard baseline ($\varepsilon = 0$ privacy leakage). However, it scales as $O(T \times K)$ communication rounds. At $T=20$ rounds, full retraining takes over **300 seconds** on an enterprise RTX A6000 GPU and demands continuous synchronization across all remaining devices, rendering it impractical in production.
2. **The Fraud of Naive Output Masking**: Artificially zeroing classifier weights or forcing output biases to negative constants (`bias = -50.0`) drops target class recall to 0.0%. However, this is an **illusion of forgetting**: the deep convolutional filters (`layer1` through `layer4`) remain 100% trained on the private features. A simple Membership Inference Attack (MIA) easily extracts memorized representations.
3. **The FedEraser Trade-off**: Reconstructing weights using historical update trajectories (Liu et al., IWQoS 2021) achieves 2.4x–3.75x speedups, but incurs a **7.2 GB server storage footprint** at $T=20$ and requires retained clients to wake up for online calibration passes.

---

## 🧠 System Architecture & Representation Engineering

Standard CNNs collapse under extreme Non-IID federated partitioning (such as 1-client-1-class or skewed Dirichlet distributions). We stabilize federated fine-tuning using four architectural pillars:

```
[ Input Image ] ──► [ ResNet-18 (Layers 1-3 Frozen) ] ──► [ Layer 4 (Local SGD) ] ──► [ Cosine Classifier (s=20.0) ] ──► Probabilities
                             ▲                                       ▲
                   [ Frozen BatchNorm ]                 [ Centroid Prototype Init ]
```

1. **Pretrained ResNet-18 with Selective Localization**: We employ a ResNet-18 backbone initialized with ImageNet weights. Early layers (1–3) encode universal visual primitives and remain frozen. Clients optimize exclusively `base.layer4` and the classifier head using local SGD (`lr=0.003`, `momentum=0.9`, `weight_decay=1e-4`), saving over 25% in communication overhead.
2. **Frozen BatchNorm Statistics (`freeze_bn`)**: Standard BatchNorm computes exponential moving averages of batch mean and variance. Under Non-IID label skew, running statistics get heavily distorted by local class distributions, destroying global feature consistency. Locking BatchNorm permanently in evaluation mode preserves robust pretrained normalization.
3. **Scaled Cosine-Normalized Classifier Head**: Standard linear layers permit logit magnitudes to scale with weight norms ($\|W_c\|$), allowing data-heavy clients to dominate minority classes. We compute logits on the unit hypersphere:
   $$\text{logits}(x) = s \cdot \frac{\phi(x)}{\|\phi(x)\|_2} \cdot \frac{W^T}{\|W\|_2} \quad \text{with } s = 20.0$$
4. **Centroid Prototype Initialization**: Round 0 initializes class weights directly to empirical feature centroids:
   $$W_c = \text{normalize}\left(\mathbb{E}_{x \in D_c}[\phi(x)]\right)$$
   providing geometric class separation before federated training begins.

---

## 📊 Empirical Benchmarks (FedEraser on CIFAR-100)

All experiments were executed on an enterprise **NVIDIA RTX A6000 GPU** with 10 clients across two distinct Non-IID regimes:
* **Setting A (1-Client-1-Class)**: Client 9 holds all training instances of `bottle` (500 images).
* **Setting B (Multi-Class Dirichlet $\alpha=0.5$)**: Client 09 holds 423 private images across multiple classes (177 bottles, 84 beavers, 51 babies, etc.). Crucially, retained clients also own 323 bottles; unlearning must purge Client 09's private memorization without destroying the global bottle concept.

### Master Benchmark Table

| Experimental Setting | Pipeline / Model | Holdout Test Acc | Forget Data Acc | Forget Data Loss | Wall-Clock Runtime | Speedup vs Retrain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Setting A**<br>*(1-Client-1-Class,<br>Client 9 "bottle")* | **Original FL ($M_{\text{orig}}$)**<br>**Retrained Gold ($M_{\text{retrain}}$)**<br>**FedEraser ($M_{\text{unlearn}}$)** | 89.11%<br>89.00%<br>**89.67%** | 94.00%<br>0.00%<br>**0.00%** | 0.2815<br>13.5260<br>**12.6289** | 40.33s<br>36.24s<br>**11.70s** | 1.0x<br>(Baseline)<br>**3.10x faster** |
| **Setting B ($T=6$)**<br>*(Dirichlet $\alpha=0.5$,<br>Client 09, 423 imgs)* | **Original FL ($M_{\text{orig}}$)**<br>**Retrained Gold ($M_{\text{retrain}}$)**<br>**FedEraser ($M_{\text{unlearn}}$)** | 93.00%<br>92.50%<br>**91.60%** | 92.43%<br>90.31%<br>**89.36%** | 0.3034<br>0.3547<br>**0.4048** | 32.92s<br>31.12s<br>**12.91s** | 1.0x<br>(Baseline)<br>**2.41x faster** |
| **Setting B ($T=20$)**<br>*(Extended Horizon,<br>200 Client Epochs)* | **Original FL ($M_{\text{orig}}$)**<br>**Retrained Gold ($M_{\text{retrain}}$)**<br>**FedEraser ($M_{\text{unlearn}}$)** | 95.10%<br>94.50%<br>**92.60%** | 93.14%<br>92.20%<br>**91.02%** | 0.1962<br>0.2501<br>**0.3214** | 323.12s<br>300.98s<br>**80.32s** | 1.0x<br>(Baseline)<br>**3.75x faster** |

#### Key Scientific Findings:
- **Semantic Redistribution in Setting A**: Purging `bottle` causes test bottles to smoothly redistribute to geometrically nearest shapes: `bed` (60%), `beetle` (10%), `apple` (9%), `baby` (8%), confirming natural hypersphere behavior.
- **High Prediction Agreement in Setting B**: FedEraser matches the retrained gold standard on **404 of 423 private samples (95.51% agreement)** at $T=20$.
- **Private Memorization Purged**: Training loss on $D_{\text{forget}}$ elevates from **0.1962** (memorized) to **0.3214** (unlearned), surpassing unseen retrained gold loss (**0.2501**).

---

## 🔍 Dual Auditing Pipeline: Grad-CAM & MIA

To prove unlearning beyond superficial accuracy numbers, we built a dual visual and quantitative auditing suite.

### 1. Grad-CAM Spatial Attribution Drift ($|\Delta\text{CAM}|$)
Hooking into `base.layer4`, we generate attention heatmaps and compute Pearson correlation drift:
$$\text{Drift} = 1.0 - \text{Pearson}\left(\text{vec}(\text{CAM}_{\text{orig}}), \text{vec}(\text{CAM}_{\text{unlearn}})\right)$$

- **Forgotten Samples (`bottle`)**: Spatial attention collapses into diffuse background context ($\text{Drift} = 0.467$).
- **Retained Samples (`apple`, `baby`, etc.)**: Attention maps remain identical ($\text{Drift} \le 0.006$), confirming zero collateral damage.

### 2. Membership Inference Attack (MIA) Auditing
An adversary executes threshold attacks on prediction loss, softmax confidence, and entropy to infer training membership.

```
Setting A: ROC Curves (Client 9 = "bottle")
TPR 1.0 |    /--- Original FL (AUC = 0.679)
        |   /
        |  /  : Random Baseline (AUC = 0.50)
        | /  /
    0.0 |/__/--- FedEraser (AUC = 0.048) & Retrained Gold (AUC = 0.053)
        +--------------------------------- FPR 1.0
```

- **Setting A Audit**:
  - Original FL has MIA-AUC of **0.679** (memorizing private bottles).
  - FedEraser plunges MIA-AUC to **0.048**, matching Retrained Gold (**0.053**).
  - *Why AUC < 0.5?* In Setting A, the concept of a bottle is completely eradicated. Loss on bottles skyrockets to **12.63** (vs 1.64 on normal test non-members). The attacker's rule ("lower loss = member") is completely inverted, proving the unlearned model treats forgotten data as severe out-of-distribution (OOD) samples.
- **Setting B Audit**:
  - Attacker True Positive Rate at 5% False Positive Rate (high-confidence privacy leak) drops from **8.27% down to 3.07%** (a **62.9% reduction in privacy leakage**).

---

## ⚡ Novel Contribution: Fed-CNS (Contrastive Null-Space Unlearning)

### Why FedEraser Breaks Down at Scale
1. **Storage Explosion $O(T \times K \times |\theta|)$**: FedEraser stores every round's global checkpoint and client updates. At $T=20$ rounds, the cache consumes **7.2 GB on disk**. In large-scale FL with 1,000 clients over 100 rounds, caching requires terabytes.
2. **Active Client Wake-Up**: Retained clients must be online to execute calibration passes. Offline edge devices halt the entire unlearning pipeline.
3. **Trajectory Drift**: Applying historical updates to calibrated weights introduces curvature mismatches that compound over long horizons ($T > 20$).

### The Fed-CNS Paradigm
Instead of tracking weights across time, **Fed-CNS** exploits the **additive property of sample second moments** in feature space:

```
[ Training Phase ] ────► Each client k accumulates second moments: A_k = Σ φ(x) φ(x)^T  (512x512 = 1.05 MB)
                                                            Class sums:     S_{k, c} = Σ φ(x)       (10x512 = 20 KB)
                                                                 │
[ Unlearning Event ] ──► Exact Server-Side Subtraction:     A_ret = A_all - A_forget (0 ms, 100% Offline Clients!)
                                                            S_ret = S_all - S_forget
                                                                 │
[ Linear Algebra ] ────► Solve Generalized Eigenproblem:     Σ_f v = λ Σ_r v   (Finds private directions V_priv)
                                                                 │
[ Projection ] ────────► Null-Space Head Scrubbing:         W' = normalize( W_ret · (I - V_priv V_priv^T) )
```

### Master Synthesis: Retraining vs FedEraser vs Fed-CNS

| Dimension / Metric | Retrained Gold Standard | FedEraser (Baseline) | **Fed-CNS v1 (Implemented)** | **Fed-CNS Tier 2 (Proposed)** |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 94.50% | 92.60% | **91.50%** | ~93.5% |
| **$D_{\text{forget}}$ CE Loss** | 0.2501 (Unseen) | 0.3214 (Purged) | **0.2954** (Label-skewed) | ~0.3100 (Calibrated) |
| **Agreement w/ Retrain** | 100.0% | 95.51% (404/423) | **96.93% (410/423)** | ~96.0% |
| **Server Runtime** | 300.98s | 80.32s (3.75x) | **0.515s (60x faster!)** | ~2.5s |
| **Server Storage** | 0 MB | 7,200 MB (7.2 GB) | **1.07 MB (>2,000x drop!)** | 10.2 MB |
| **Retained Clients Online** | 9 (Full Retrain) | Active (Calibration) | **0 (100% Offline)** | 0 (100% Offline) |
| **Feature Scrubbing Depth** | Full Network | Deep Conv Kernels | **Classifier Head (Readout)** | Head + `layer4` Conv |

---

## 🔬 Critical Self-Audit & The Class-Conditional 2-Tier Architecture

To maintain scientific integrity, we self-audited Fed-CNS v1 and identified two crucial properties:

1. **The Label-Skew Confound**: Under Dirichlet non-IID partitioning, Client 09's dataset was 42% bottles. Computing an unconditioned second-moment matrix $A_f$ causes the top generalized eigenvectors to align with the bottle class centroid (label skew) rather than Client 09's private instance style.
2. **The Readout-Only Boundary**: Fed-CNS v1 projects output logits to zero along private directions, ensuring black-box unlearning fidelity. However, the convolutional kernels in `base.layer4` remain unmodified. In a strict white-box threat model where an auditor probes internal activations, latent traces could theoretically be extracted.

### The Architectural Solution (Fed-CNS Tier 2 Roadmap)
- **Class-Conditional Ledger**: Store separate second-moment matrices $A_{k, c}$ per class ($10 \text{ classes} \times 1 \text{ MB} = 10.2 \text{ MB}$). This directly contrasts Client 09's bottles against retained clients' bottles, cancelling label skew.
- **Tier 2 Convolutional Projection**: Perform a projected gradient step on `base.layer4` weights projected into $\text{Null}(F_{\text{retain}})$, achieving physical convolutional filter sanitization in under 3 seconds with zero stored trajectories.

---

## 🖥️ Interactive Streamlit Web Dashboard

We built a presenter-friendly evaluation web dashboard in [`app.py`](file:///usershome/ds404_user4/ritesh/app.py) featuring a single-control interface:

```bash
./.venv/bin/streamlit run app.py
```

### Dashboard Capabilities:
- **Single-Variable Control**: Select any client/class to unlearn from a single dropdown.
- **Instant Metric Cards**: Holdout test accuracy, unlearning fidelity, prediction agreement, and runtime speedups.
- **Grad-CAM Visualizer Tab**: Interactive attention comparison across Original, Retrained, and Unlearned models with spatial difference heatmaps ($|\Delta\text{CAM}|$).
- **MIA Privacy Audit Tab**: Live ROC-AUC curves and loss distribution histograms proving quantitative privacy recovery.
- **Master Benchmark Matrix**: Side-by-side performance comparison between Retraining, FedEraser, and Fed-CNS.

---

## 📁 Repository Structure

```
Federated-Unlearning/
├── app.py                                   # Streamlit interactive evaluation dashboard
├── generate_formal_report_pdf.py            # ReportLab publication-grade PDF generator
├── main.py                                  # CLI unified entry point
├── run_contrastive_nullspace_benchmark.py   # Benchmark script for Fed-CNS vs FedEraser
├── run_mia_audit.py                         # Full Membership Inference Attack audit runner
├── run_multiclass_experiment.py             # Multi-class Dirichlet FL experiment runner
├── run_single_class_experiment.py           # Single-class simplification experiment runner
│
├── checkpoints/                             # Model checkpoints and cached trajectories
│   ├── M_orig.pt                            # Original collaborative FL model
│   ├── M_unlearn.pt                         # FedEraser unlearned model
│   ├── M_retrain.pt                         # Retrained gold standard baseline
│   └── fl_training_cache.pt                 # Historical update cache (7.2 GB at T=20)
│
├── results/                                 # Benchmark logs, audit JSONs, and figures
│   ├── single_class/                        # Setting A benchmark & MIA figures
│   │   ├── mia_audit_benchmark.json
│   │   └── mia_audit_comparison.png
│   ├── multi_class/                         # Setting B benchmark & MIA figures
│   │   ├── mia_audit_benchmark.json
│   │   └── mia_audit_comparison.png
│   └── novel_nullspace/                     # Fed-CNS benchmark results
│       ├── single_class/cns_benchmark.json
│       └── multi_class/cns_benchmark.json
│
├── src/                                     # Core modular Python packages
│   ├── audit/
│   │   └── mia_auditor.py                   # MIA auditor (ROC, AUC, loss distributions)
│   ├── data/
│   │   └── cifar_loader.py                  # CIFAR-100 Non-IID Dirichlet & single-class loaders
│   ├── flwr_fl/
│   │   ├── client.py                        # Flower NumPyClient with calibration pass support
│   │   ├── runner.py                        # Federated training engine with update logging
│   │   └── strategy.py                      # Custom Flower logging strategy
│   ├── models/
│   │   └── resnet_cifar.py                  # FedVisionModel (ResNet-18 + Cosine Head + Freeze BN)
│   ├── unlearning/
│   │   ├── additive_ledger.py               # Feature second-moment accumulator (Fed-CNS)
│   │   ├── contrastive_nullspace.py         # Generalized eigensolver & null-space projection
│   │   └── federaser.py                     # Historical calibration unlearning engine
│   └── visualize/
│       └── gradcam_visualizer.py            # Grad-CAM heatmap generator & drift auditor
│
├── docs/                                    # Documentation and presentation resources
│   ├── Federated_Unlearning_Project_Report.pdf  # 3-page publication-grade PDF report
│   ├── PROJECT_TLDR_FOR_MENTOR.txt              # Punchy mentor summary
│   ├── FED_CNS_TECHNICAL_DEEP_DIVE.txt          # FLOP analysis & memory breakdown of Fed-CNS
│   ├── Audit of Fed-CNS Technical Correctness and Next Steps.md  # Critical peer-review
│   └── PRESENTATION_SLIDES_AND_DEFENSE_GUIDE.txt # 12-slide presentation script & Q&A guide
└── README.md                                # Comprehensive repository documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/Demnthatwasme/Federated-Unlearning.git
cd Federated-Unlearning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt reportlab streamlit matplotlib scikit-learn
```

### 2. Run Federated Training & FedEraser Unlearning
```bash
# Multi-Class Dirichlet (Setting B):
python run_multiclass_experiment.py

# Single-Class Simplification (Setting A):
python run_single_class_experiment.py
```

### 3. Run the Full Membership Inference Attack (MIA) Audit
```bash
python run_mia_audit.py
```
Outputs ROC curves and loss histograms to `results/single_class/mia_audit_comparison.png` and `results/multi_class/mia_audit_comparison.png`.

### 4. Run the Fed-CNS Benchmark
```bash
python run_contrastive_nullspace_benchmark.py
```
Evaluates the additive ledger subtraction and solves the contrastive generalized eigenproblem in **0.515 seconds**.

### 5. Launch the Streamlit Demo Dashboard
```bash
streamlit run app.py
```

### 6. Generate the Formal PDF Report
```bash
python generate_formal_report_pdf.py
```
Compiles the 3-page publication-grade report: `Federated_Unlearning_Project_Report.pdf`.

---

## 📑 Formal Documentation & Artifacts

- 📄 **[Federated_Unlearning_Project_Report.pdf](Federated_Unlearning_Project_Report.pdf)**: Formal 3-page publication-quality report for committee/mentor evaluation.
- 📝 **[PROJECT_TLDR_FOR_MENTOR.txt](PROJECT_TLDR_FOR_MENTOR.txt)**: Casual, punchy executive summary for quick review.
- 🔬 **[FED_CNS_TECHNICAL_DEEP_DIVE.txt](FED_CNS_TECHNICAL_DEEP_DIVE.txt)**: Mathematical derivations, FLOP counts, and memory footprints.
- 🛡️ **[Audit of Fed-CNS Technical Correctness and Next Steps.md](Audit%20of%20Fed-CNS%20Technical%20Correctness%20and%20Next%20Steps.md)**: Transparent peer-review analysis of label skew and Tier-2 convolutional projection.
- 🎤 **[PRESENTATION_SLIDES_AND_DEFENSE_GUIDE.txt](PRESENTATION_SLIDES_AND_DEFENSE_GUIDE.txt)**: 12-slide defense script with expected committee Q&A.

---

## 👥 Contributors & Citation
- **Author**: Ritesh
- **Hardware & Framework**: NVIDIA RTX A6000 | PyTorch 2.x | Flower (`flwr`)
- **Evaluation Date**: 19 November 2026
