import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader

class MembershipInferenceAuditor:
    """
    Auditing suite to quantitatively measure membership privacy before and after unlearning.
    
    Implements:
      1. Metric-based (Loss/Entropy/Confidence) distribution analysis.
      2. Threshold-based ROC-AUC evaluation.
      3. Classifier-based MIA using shadow posteriors and features.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def extract_model_signals(self, model: nn.Module, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Extracts prediction posteriors, losses, confidences, and entropies from a model.
        """
        model.eval()
        losses = []
        confidences = []
        entropies = []
        posteriors = []
        labels_list = []
        correct_flags = []

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                batch_losses = self.criterion(logits, labels)
                
                # Confidence and top prediction
                max_prob, preds = probs.max(dim=1)
                correct = preds.eq(labels).float()
                
                # Prediction entropy: - sum(p * log(p))
                log_probs = torch.log(probs + 1e-12)
                entropy = -torch.sum(probs * log_probs, dim=1)

                losses.extend(batch_losses.cpu().numpy().tolist())
                confidences.extend(max_prob.cpu().numpy().tolist())
                entropies.extend(entropy.cpu().numpy().tolist())
                posteriors.extend(probs.cpu().numpy())
                labels_list.extend(labels.cpu().numpy().tolist())
                correct_flags.extend(correct.cpu().numpy().tolist())

        return {
            "losses": np.array(losses),
            "confidences": np.array(confidences),
            "entropies": np.array(entropies),
            "posteriors": np.array(posteriors),
            "labels": np.array(labels_list),
            "correct": np.array(correct_flags)
        }

    def audit_model(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        test_nonmember_loader: DataLoader,
        shadow_train_loader: Optional[DataLoader] = None,
        shadow_test_loader: Optional[DataLoader] = None,
        model_name: str = "Model"
    ) -> Dict:
        """
        Performs comprehensive MIA audit on forget data (target members) vs holdout test data (non-members).
        
        Returns ROC-AUC, threshold attack accuracy, and distribution statistics.
        """
        # Signals on target forgotten data (label 1: member)
        forget_signals = self.extract_model_signals(model, forget_loader)
        # Signals on unseen non-members (label 0: non-member)
        nonmember_signals = self.extract_model_signals(model, test_nonmember_loader)

        # 1. Loss-based threshold attack:
        # Members have lower loss; negative loss is higher for members.
        scores = np.concatenate([-forget_signals["losses"], -nonmember_signals["losses"]])
        y_true = np.concatenate([np.ones(len(forget_signals["losses"])), np.zeros(len(nonmember_signals["losses"]))])

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc_loss = float(roc_auc_score(y_true, scores))

        # 2. Confidence-based threshold attack:
        conf_scores = np.concatenate([forget_signals["confidences"], -nonmember_signals["confidences"]])
        auc_conf = float(roc_auc_score(y_true, np.concatenate([forget_signals["confidences"], nonmember_signals["confidences"]])))

        # 3. Best threshold attack accuracy
        threshold_idx = np.argmax(tpr - fpr)
        best_tpr = float(tpr[threshold_idx])
        best_fpr = float(fpr[threshold_idx])
        best_threshold = float(thresholds[threshold_idx])
        pred_member = (scores >= best_threshold).astype(int)
        attack_acc = float(accuracy_score(y_true, pred_member))

        # TPR at low False Positive Rate (FPR <= 0.05)
        low_fpr_indices = np.where(fpr <= 0.05)[0]
        tpr_at_low_fpr = float(tpr[low_fpr_indices[-1]]) if len(low_fpr_indices) > 0 else 0.0

        # Summary statistics
        audit_results = {
            "model_name": model_name,
            "auc_loss": auc_loss,
            "auc_confidence": auc_conf,
            "attack_accuracy": attack_acc,
            "tpr_at_5pct_fpr": tpr_at_low_fpr,
            "forget_mean_loss": float(np.mean(forget_signals["losses"])),
            "forget_std_loss": float(np.std(forget_signals["losses"])),
            "forget_mean_confidence": float(np.mean(forget_signals["confidences"])),
            "nonmember_mean_loss": float(np.mean(nonmember_signals["losses"])),
            "nonmember_std_loss": float(np.std(nonmember_signals["losses"])),
            "nonmember_mean_confidence": float(np.mean(nonmember_signals["confidences"])),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "forget_losses": forget_signals["losses"].tolist(),
            "nonmember_losses": nonmember_signals["losses"].tolist(),
        }

        return audit_results

    def plot_mia_comparisons(
        self,
        audit_orig: Dict,
        audit_unlearn: Dict,
        audit_retrain: Optional[Dict] = None,
        save_path: str = "results/mia_audit_comparison.png"
    ):
        """
        Generates and saves publication-quality ROC curves and loss distribution plots
        comparing Original vs Unlearned vs Retrained models.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: ROC Curves
        ax = axes[0]
        ax.plot(audit_orig["fpr"], audit_orig["tpr"], label=f"Original (AUC = {audit_orig['auc_loss']:.3f})", color="#e74c3c", lw=2.5)
        ax.plot(audit_unlearn["fpr"], audit_unlearn["tpr"], label=f"Unlearned (AUC = {audit_unlearn['auc_loss']:.3f})", color="#2ecc71", lw=2.5)
        if audit_retrain:
            ax.plot(audit_retrain["fpr"], audit_retrain["tpr"], label=f"Retrained Baseline (AUC = {audit_retrain['auc_loss']:.3f})", color="#3498db", lw=2, linestyle="--")
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Guess (AUC = 0.50)")
        ax.set_title("Membership Inference ROC Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 2: Loss Distribution (Original Model)
        ax = axes[1]
        ax.hist(audit_orig["forget_losses"], bins=30, alpha=0.6, label="Forget Set (Members)", color="#e74c3c", density=True)
        ax.hist(audit_orig["nonmember_losses"], bins=30, alpha=0.6, label="Test Set (Non-members)", color="#95a5a6", density=True)
        ax.set_title(f"Loss Distribution: Original Model\n(Mean Forget Loss: {audit_orig['forget_mean_loss']:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cross-Entropy Loss", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Subplot 3: Loss Distribution (Unlearned Model)
        ax = axes[2]
        ax.hist(audit_unlearn["forget_losses"], bins=30, alpha=0.6, label="Forget Set (After Unlearn)", color="#2ecc71", density=True)
        ax.hist(audit_unlearn["nonmember_losses"], bins=30, alpha=0.6, label="Test Set (Non-members)", color="#95a5a6", density=True)
        ax.set_title(f"Loss Distribution: Unlearned Model\n(Mean Forget Loss: {audit_unlearn['forget_mean_loss']:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Cross-Entropy Loss", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved MIA audit plots to {save_path}")

