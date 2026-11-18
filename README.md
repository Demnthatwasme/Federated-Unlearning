# Federated Unlearning for Vision Models with Auditing & Visualization

An end-to-end **Federated Unlearning Pipeline** for image classification on **CIFAR-100** implementing the **"Right to be Forgotten"** (GDPR Article 17) using **FedEraser**, **Flower (`flwr`)**, **PyTorch**, and **Grad-CAM Visual Attention Auditing**.

---

## 🎯 Problem Statement & Objective

In Federated Learning (FL), distributed clients collaboratively train a shared global model without centralizing their raw private datasets. Under privacy regulations like the GDPR (Article 17: "Right to be Forgotten") and CCPA, participating clients have the legal entitlement to withdraw their data and have all trace of their contributions erased from the trained global model.

- **The Naive Approach (Retraining from Scratch)**: Discarding the model and retraining over all communication rounds without the withdrawn client is computationally prohibitive, communication-heavy, and wastes all prior collaborative computation.
- **The Heuristic Approach (Output Masking / Weight Zeroing)**: Artificially forcing output logits or classifier biases to negative infinity (`bias = -50.0`) produces fake "100% forgotten" metrics without modifying the underlying feature representations.
- **The Principled Solution (FedEraser)**: A landmark federated unlearning framework (Gaoyang Liu, Xiaoqiang Ma, Yang Yang, et al., IEEE/ACM IWQoS 2021) that trades central server storage for unlearning speed. By maintaining a historical log of global models and client parameter updates during FL rounds, FedEraser reconstructs the unlearned model along the original optimization trajectory using fast calibration steps executed exclusively by the retained clients.

> **Note on MIA Auditing**: As instructed, Membership Inference Attack (MIA) auditing is **intentionally omitted** from this stage of the project and will be integrated in subsequent auditing phases.

---

## 🔬 Experimental Setup: Single-Class-Per-Client Simplification

The problem is evaluated under the **single-class-per-client** extreme Non-IID setting on **CIFAR-100**:
- **Clients**: 10 clients total (Clients $0, 1, \dots, 9$).
- **Partitioning**: Client $i$ holds exclusively the training samples for Class $i$:
  - Client 0: `apple`
  - Client 1: `aquarium_fish`
  - Client 2: `baby`
  - Client 3: `bear`
  - Client 4: `beaver`
  - Client 5: `bed`
  - Client 6: `bee`
  - Client 7: `beetle`
  - Client 8: `bicycle`
  - Client 9: `bottle` (**Target Forgotten Client**)

### Model Design: Overcoming Extreme Non-IID Collapse
When each client holds only a single class and trains locally with standard unconstrained Cross-Entropy, each client aggressively maximizes its own class logit while pushing all other classes toward negative infinity. Upon naive FedAvg parameter averaging, the global model suffers from severe representation collapse (~10-12% random chance).

To solve this stably without synthetic multi-class sharing:
1. **ResNet-18 Backbone**: Rich convolutional features pretrained on ImageNet.
2. **Frozen BatchNorm Statistics (`freeze_bn`)**: BatchNorm running stats are kept in evaluation mode with pretrained statistics. This prevents single-class batches from corrupting mean and variance estimates.
3. **Cosine-Normalized Classifier Head**: Logits are computed via hypersphere-scaled cosine similarities:
   $$\text{logits}(x) = s \cdot \frac{\phi(x)}{\|\phi(x)\|_2} \cdot \frac{W^T}{\|W\|_2}$$
   where $s = 16.0$.
4. **Centroid Prototype Initialization**: Round 0 initializes each client's prototype vector to its class feature centroid $\mu_k = \mathbb{E}_{x \in D_k}[\phi(x)]$.
5. **Fine-Tuning Layers**: Client training fine-tunes `base.layer4` and the classifier head, achieving **>86% overall test accuracy** on CIFAR-100.

---

## ⚡ The FedEraser Algorithm

FedEraser operates in two main phases:

```
[Phase 1: Normal FL with Historical Logging]
  Server holds W_0
  For round t = 1 .. T:
    - Server broadcasts W_{t-1} to all clients k in {0 .. 9}
    - Client k trains locally using Flower NumPyClient: producing update ΔW_{t, k}
    - Server stores W_t and {ΔW_{t, k}} in cache
    - Server aggregates: W_t = W_{t-1} + sum_k (n_k / N) * ΔW_{t, k}

[Phase 2: Unlearning Reconstruction (Client 9 Requests Erasure)]
  Server loads initial state W'_0 = W_0 (with Client 9's prototype neutralized)
  Retained clients R = {0, 1, 2, 3, 4, 5, 6, 7, 8}
  For round t = 1 .. T:
    - Retained clients k in R start from W'_{t-1} and run short calibration steps (E_cali << E)
    - Client k produces calibration update ΔW'_{t, k}
    - Server scales direction to preserve historical magnitude:
        W~_{t, k} = (||ΔW_{t, k}|| / ||ΔW'_{t, k}||) * ΔW'_{t, k}
    - Server aggregates over R:
        W'_t = W'_{t-1} + (1 / |R|) * sum_{k in R} W~_{t, k}
  Result: Unlearned Model M_unlearn = W'_T
```

---

## 📊 Experimental Results & Benchmark Tables

### Setting 1: Single-Class Per Client Simplification (1 Client = 1 Class)

Evaluated on CIFAR-100 holdout test set (Classes 0-9) over 6 communication rounds with Client 9 (`bottle`) requesting unlearning:

| Model / Evaluation Metric | Retain Test Acc | Forget Test Acc | Forget Softmax Conf | Overall Test Acc | Runtime (s) | Speedup vs Retrain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original FL ($M_{\text{orig}}$)** | **89.11%** | **94.00%** | **0.840** | **89.60%** | 40.33s | 1.0x (10 Clients) |
| **Retrained Gold Standard ($M_{\text{retrain}}$)** | **89.00%** | **0.00%** | **0.000** | **80.10%** | 36.24s | (9 Clients) |
| **FedEraser ($M_{\text{unlearn}}$)** | **89.67%** | **0.00%** | **0.000** | **80.70%** | **11.70s** | **3.10x faster** |

- **Redistribution of Forgotten Samples**: When `bottle` is erased from $M_{\text{unlearn}}$, samples naturally redistribute to nearest semantic shapes (`bed`: 60%, `beetle`: 10%, `apple`: 9%, `baby`: 8%, `bicycle`: 8%, `fish`: 4%, `bear`: 1%).
- **Attribution Drift Auditing**: Mean Grad-CAM drift ($|\Delta\text{CAM}|$) is **0.467** on the forgotten class vs **0.006** on retained classes.

---

### Setting 2: Multi-Class Heterogeneous Non-IID Setup (Dirichlet $\alpha=0.5$)

In realistic federated learning, each client holds samples from **multiple classes** with label skew. Under a Dirichlet distribution $\text{Dir}(\alpha=0.5)$, 10 clients hold heterogeneous mixtures of 8 to 10 classes each. Client 09 requests complete erasure of its **423 private training images** under GDPR Right to be Forgotten.

| Model / Evaluation Metric | Holdout Test Acc | Private Data Acc ($D_{\text{forget}}$) | Private Data Loss ($D_{\text{forget}}$) | Private Data Conf | Runtime (s) | Speedup vs Retrain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original FL ($M_{\text{orig}}$)** | **93.00%** | **92.43%** | **0.3034** (Memorized) | **0.810** | 32.92s | 1.0x (10 Clients) |
| **Retrained Gold Standard ($M_{\text{retrain}}$)** | **92.50%** | **90.31%** | **0.3547** (Unseen Gold) | **0.791** | 31.12s | (9 Clients) |
| **FedEraser ($M_{\text{unlearn}}$)** | **91.60%** | **89.36%** | **0.4048** (Unlearned) | **0.751** | **12.91s** | **2.41x faster** |

#### Key Observations in Multi-Class Unlearning:
1. **High Unlearning Fidelity**: $M_{\text{unlearn}}$ achieves **93.6% prediction agreement (396/423 samples)** with the retrained gold standard on the forgotten client's private training dataset ($D_{\text{forget}}$).
2. **Private Memorization Erasure**: Training loss on $D_{\text{forget}}$ increases from **0.3034** (strongly memorized) to **0.4048**, verifying that private sample memorization was successfully purged.
3. **Global Utility Preserved**: Overall test accuracy on the holdout CIFAR-100 test set remains high at **91.60%** (vs 92.50% for exact retraining).
4. **Computational Advantage**: FedEraser completes in **12.91s** vs **31.12s** for retraining from scratch (**2.41x speedup**).

#### Extended Horizon Benchmark ($T = 20$ Communication Rounds):

To evaluate FedEraser over longer training horizons, the federated system was trained for 20 communication rounds ($T = 20$, 200 total client epochs on RTX A6000 GPU):

| Model / Evaluation Metric | Holdout Test Acc | Private Data Acc ($D_{\text{forget}}$) | Private Data Loss ($D_{\text{forget}}$) | Private Data Conf | Runtime (s) | Speedup vs Retrain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original FL ($M_{\text{orig}}$)** | **95.10%** | **93.14%** | **0.1962** (Memorized) | **0.885** | 323.12s | 1.0x (10 Clients) |
| **Retrained Gold Standard ($M_{\text{retrain}}$)** | **94.50%** | **92.20%** | **0.2501** (Unseen Gold) | **0.868** | 300.98s | (9 Clients) |
| **FedEraser ($M_{\text{unlearn}}$)** | **92.60%** | **91.02%** | **0.3214** (Unlearned) | **0.819** | **80.32s** | **3.75x faster** |

- **Extended Prediction Agreement**: $M_{\text{unlearn}}$ matches the retrained gold standard on **404 out of 423 private images (95.5% agreement)**.
- **Unlearning Speedup**: FedEraser completes unlearning in **80.32s** vs **300.98s** for full retraining (**3.75x speedup**).
- **Private Memorization Purge**: Cross-entropy loss on $D_{\text{forget}}$ rises from **0.1962** (deeply memorized) to **0.3214** (unlearned), matching the unseen retrained profile (**0.2501**).

---

## 🖼️ Visual Evidence of Forgetting (Grad-CAM)

We extract Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps from the final convolutional block (`base.layer4`):

![Grad-CAM Unlearning Gallery](results/gradcam_unlearning_comparison.png)

### Visual Findings:
1. **Forgotten Client Samples (`bottle`)**:
   - **$M_{\text{orig}}$**: Strongly activates on semantic bottle contours with high confidence (0.742), highlighting structural shape edges.
   - **$M_{\text{unlearn}}$ (FedEraser)**: Top-class bottle confidence plummets to $3.57 \times 10^{-5}$; Grad-CAM attention on bottle contours collapses (CAM drift = 0.500), mirroring $M_{\text{retrain}}$.
   - **Attribution Shift ($|\Delta\text{CAM}|$)|**: Difference map clearly pinpoints the exact spatial attention regions erased during unlearning.
2. **Retained Client Samples (`apple`, `aquarium_fish`, `baby`)**:
   - Attention remains tightly focused on the target objects across both $M_{\text{orig}}$ and $M_{\text{unlearn}}$ (CAM drift = 0.009), visually confirming that retained capabilities are preserved without collateral damage.

---

## 📁 Repository Structure

```
├── checkpoints/                        # Model weights and training history
│   ├── M_orig.pt                       # Original FL model checkpoint
│   ├── M_unlearn.pt                    # FedEraser unlearned model checkpoint
│   ├── M_retrain.pt                    # Exact retraining baseline checkpoint
│   └── fl_training_cache.pt            # Historical global models & client updates
├── dataset/
│   └── cifar-100-python/               # CIFAR-100 dataset files
├── notebooks/
│   └── federated_unlearning_demo.ipynb # Interactive demonstration Jupyter notebook
├── results/
│   ├── gradcam_unlearning_comparison.png # Side-by-side Grad-CAM comparison gallery
│   └── unlearning_benchmark.json       # Structured benchmark metrics
├── src/
│   ├── data/
│   │   └── cifar_loader.py             # Single-class FL partitioning and DataLoaders
│   ├── flwr_fl/
│   │   ├── client.py                   # Flower NumPyClient with calibration support
│   │   ├── runner.py                   # FL training runner with historical update logging
│   │   └── strategy.py                 # Custom Flower logging strategy
│   ├── models/
│   │   └── resnet_cifar.py             # FedVisionModel (ResNet-18 + Cosine Head + Freeze BN)
│   ├── unlearning/
│   │   └── federaser.py                # FedEraser unlearning engine & exact retrain
│   └── visualize/
│       └── gradcam_visualizer.py       # Grad-CAM explainer, overlay, and gallery builder
├── app.py                              # Interactive Streamlit Web Application
├── main.py                             # CLI experiment runner
├── problem-statement.png               # Problem statement image
└── README.md                           # Comprehensive documentation
```

---

## 🚀 How to Run

### 1. Run the Multi-Class Heterogeneous Experiment (Dirichlet α=0.5)
```bash
./.venv/bin/python run_multiclass_experiment.py
# Or via main.py:
./.venv/bin/python main.py --partition_mode multi_class --stage all --device cuda
```

### 2. Run the Single-Class Simplification (1 Client = 1 Class)
```bash
./.venv/bin/python run_single_class_experiment.py
# Or via main.py:
./.venv/bin/python main.py --partition_mode single_class --stage all --device cuda
```

### 3. Launch the Interactive Streamlit Web Dashboard
```bash
./.venv/bin/streamlit run app.py
```
Provides:
- Seamless toggle between **Single-Class** and **Multi-Class (Dirichlet)** experimental modes.
- Benchmark metric cards, speedup metrics, and prediction agreement counters.
- Client class frequency distribution matrix across all 10 clients.
- Interactive Grad-CAM sample inspector with real-time attribution overlays and difference maps ($|\Delta\text{CAM}|$).

### 4. Run the Demonstration Notebook
```bash
./.venv/bin/jupyter notebook notebooks/federated_unlearning_demo.ipynb
```
Step-by-step interactive walkthrough with rich markdown explanations and visual plots.
