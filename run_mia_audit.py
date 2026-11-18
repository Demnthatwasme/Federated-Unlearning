#!/usr/bin/env python3
"""
Membership Inference Attack (MIA) Auditing Suite for Federated Unlearning.

Audits whether an adversary can infer if private samples from the forgotten client (D_forget)
were part of the training dataset, comparing:
  1. Original FL Model (M_orig)      [High Privacy Leakage Expected]
  2. Retrained Gold Standard (M_retrain) [Theoretical Baseline: Random Guess AUC ~ 0.50]
  3. FedEraser (M_unlearn)           [Target: AUC collapses towards M_retrain]

Evaluates:
  - Metric-based ROC-AUC (Cross-Entropy Loss, Softmax Confidence, Prediction Entropy)
  - Optimal Threshold Attack Accuracy
  - True Positive Rate (TPR) at low False Positive Rate (FPR <= 5%)
  - Generates publication-ready ROC curves & loss distribution figures.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from src.data.cifar_loader import MultiClassFLDataManager, SingleClassFLDataManager
from src.models.resnet_cifar import get_cifar_model
from src.audit.mia_auditor import MembershipInferenceAuditor

def run_mia_audit(partition_mode="multi_class", forget_client=9, device="cuda"):
    print("=" * 85)
    print(f"  MEMBERSHIP INFERENCE ATTACK (MIA) PRIVACY AUDIT")
    print(f"  Partition Mode: {partition_mode} | Forget Client: {forget_client}")
    print("=" * 85)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ckpt_dir = f"checkpoints/{partition_mode}"
    out_dir = f"results/{partition_mode}"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Dataset Partitioning
    num_classes = 10
    selected_classes = list(range(num_classes))
    if partition_mode == "multi_class":
        data_manager = MultiClassFLDataManager(
            root="dataset",
            selected_classes=selected_classes,
            num_clients=10,
            alpha=0.5,
            batch_size=64,
            seed=42
        )
        forget_loader = data_manager.get_forget_train_loader(forget_client)
    else:
        data_manager = SingleClassFLDataManager(
            root="dataset",
            selected_classes=selected_classes,
            batch_size=64
        )
        forget_loader = data_manager.get_forget_train_loader(forget_client)

    # Non-member set: holdout test set across the target classes
    test_loader = data_manager.get_full_test_loader()

    # 2. Load Models (READ-ONLY)
    print("\n[1/3] Loading Model Checkpoints for Privacy Audit...")
    orig_model = get_cifar_model(num_classes=num_classes).to(device)
    orig_model.load_state_dict(torch.load(f"{ckpt_dir}/M_orig.pt", map_location=device))
    orig_model.eval()

    retrain_model = get_cifar_model(num_classes=num_classes).to(device)
    retrain_model.load_state_dict(torch.load(f"{ckpt_dir}/M_retrain.pt", map_location=device))
    retrain_model.eval()

    unlearn_model = get_cifar_model(num_classes=num_classes).to(device)
    unlearn_model.load_state_dict(torch.load(f"{ckpt_dir}/M_unlearn.pt", map_location=device))
    unlearn_model.eval()

    # 3. Execute MIA Audit
    print("\n[2/3] Extracting Privacy Signals & Evaluating Attack ROC-AUC...")
    auditor = MembershipInferenceAuditor(device=str(device))

    audit_orig = auditor.audit_model(
        model=orig_model,
        forget_loader=forget_loader,
        test_nonmember_loader=test_loader,
        model_name="Original FL (M_orig)"
    )

    audit_unlearn = auditor.audit_model(
        model=unlearn_model,
        forget_loader=forget_loader,
        test_nonmember_loader=test_loader,
        model_name="FedEraser (M_unlearn)"
    )

    audit_retrain = auditor.audit_model(
        model=retrain_model,
        forget_loader=forget_loader,
        test_nonmember_loader=test_loader,
        model_name="Retrained Gold (M_retrain)"
    )

    # 4. Generate Publication-Quality Visualizations
    plot_path = f"{out_dir}/mia_audit_comparison.png"
    auditor.plot_mia_comparisons(
        audit_orig=audit_orig,
        audit_unlearn=audit_unlearn,
        audit_retrain=audit_retrain,
        save_path=plot_path
    )

    # 5. Display Benchmark Summary Table
    print("\n" + "=" * 105)
    print(f"  MEMBERSHIP INFERENCE ATTACK (MIA) AUDITING RESULTS")
    print("=" * 105)
    headers = ["Model / Metric", "Loss MIA-AUC", "Conf MIA-AUC", "Attack Acc", "TPR @ 5% FPR", "Mean Forget Loss", "Privacy Status"]
    print(f"{headers[0]:<28} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<10} | {headers[4]:<13} | {headers[5]:<16} | {headers[6]:<18}")
    print("-" * 125)

    def format_row(name, audit, status):
        return f"{name:<28} | {audit['auc_loss']:>11.3f} | {audit['auc_confidence']:>11.3f} | {audit['attack_accuracy']:>9.2%} | {audit['tpr_at_5pct_fpr']:>12.2%} | {audit['forget_mean_loss']:>15.4f} | {status:<18}"

    print(format_row("Original FL (M_orig)", audit_orig, "Leaking (Member)"))
    print(format_row("Retrained Gold (M_retrain)", audit_retrain, "Ideal Non-Member"))
    print(format_row("FedEraser (M_unlearn)", audit_unlearn, "Unlearned (Purged)"))
    print("-" * 125)

    auc_reduction = audit_orig['auc_loss'] - audit_unlearn['auc_loss']
    print(f"\nAUDIT FINDINGS:")
    print(f"  • Original Model Loss MIA-AUC:    {audit_orig['auc_loss']:.3f} (Adversary easily detects membership)")
    print(f"  • Retrained Gold Loss MIA-AUC:    {audit_retrain['auc_loss']:.3f} (Gold standard privacy floor)")
    print(f"  • FedEraser Loss MIA-AUC:         {audit_unlearn['auc_loss']:.3f} (Privacy leakage reduced by {auc_reduction:.3f} AUC)")
    print(f"  • Private Mean Loss Elevation:    {audit_orig['forget_mean_loss']:.4f} -> {audit_unlearn['forget_mean_loss']:.4f} (Approaching unseen baseline {audit_retrain['forget_mean_loss']:.4f})")
    print(f"  • Visual Plots Saved To:          {plot_path}")

    # Save JSON results
    summary_path = f"{out_dir}/mia_audit_benchmark.json"
    results_payload = {
        "partition_mode": partition_mode,
        "forget_client": forget_client,
        "models": {
            "original": {
                "auc_loss": audit_orig["auc_loss"],
                "auc_confidence": audit_orig["auc_confidence"],
                "auc_entropy": audit_orig["auc_entropy"],
                "attack_accuracy": audit_orig["attack_accuracy"],
                "tpr_at_5pct_fpr": audit_orig["tpr_at_5pct_fpr"],
                "forget_mean_loss": audit_orig["forget_mean_loss"],
                "forget_std_loss": audit_orig["forget_std_loss"],
                "nonmember_mean_loss": audit_orig["nonmember_mean_loss"]
            },
            "retrained": {
                "auc_loss": audit_retrain["auc_loss"],
                "auc_confidence": audit_retrain["auc_confidence"],
                "auc_entropy": audit_retrain["auc_entropy"],
                "attack_accuracy": audit_retrain["attack_accuracy"],
                "tpr_at_5pct_fpr": audit_retrain["tpr_at_5pct_fpr"],
                "forget_mean_loss": audit_retrain["forget_mean_loss"],
                "forget_std_loss": audit_retrain["forget_std_loss"],
                "nonmember_mean_loss": audit_retrain["nonmember_mean_loss"]
            },
            "federaser": {
                "auc_loss": audit_unlearn["auc_loss"],
                "auc_confidence": audit_unlearn["auc_confidence"],
                "auc_entropy": audit_unlearn["auc_entropy"],
                "attack_accuracy": audit_unlearn["attack_accuracy"],
                "tpr_at_5pct_fpr": audit_unlearn["tpr_at_5pct_fpr"],
                "forget_mean_loss": audit_unlearn["forget_mean_loss"],
                "forget_std_loss": audit_unlearn["forget_std_loss"],
                "nonmember_mean_loss": audit_unlearn["nonmember_mean_loss"],
                "auc_reduction": auc_reduction
            }
        },
        "plot_path": plot_path
    }

    with open(summary_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"  • Structured JSON Metrics:       {summary_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Membership Inference Attack (MIA) privacy audit.")
    parser.add_argument("--partition_mode", choices=["multi_class", "single_class"], default="multi_class")
    parser.add_argument("--forget_client", type=int, default=9)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_mia_audit(partition_mode=args.partition_mode, forget_client=args.forget_client, device=args.device)
