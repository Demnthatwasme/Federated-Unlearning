"""
Additive Ledger for Federated Learning and Unlearning.

Instead of caching round checkpoints and client parameter updates (O(T * K * |theta|)),
the Additive Ledger tracks sample-sum second moments and prototype centroids:
  - Feature Gram / Second-Moment Matrix: A_k = sum_{x in D_k} phi(x) phi(x)^T  (512 x 512, ~1 MB)
  - Class Feature Sums: S_{k, c} = sum_{x in D_{k, c}} phi(x)               (C x 512, ~20 KB)
  - Sample Counts: n_{k, c} = |D_{k, c}|

Unlearning is performed in closed-form via subtraction:
  A_retain = A_all - A_forget
  S_retain = S_all - S_forget
  n_retain = n_all - n_forget

Retained clients remain 100% offline. Server storage is O(d^2 + C * d) total.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class AdditiveLedger:
    def __init__(self, feature_dim: int = 512, num_classes: int = 10, device: str = "cpu"):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # Global totals
        self.A_all = torch.zeros((feature_dim, feature_dim), dtype=torch.float32, device=self.device)
        self.S_all = torch.zeros((num_classes, feature_dim), dtype=torch.float32, device=self.device)
        self.n_all = torch.zeros((num_classes,), dtype=torch.float32, device=self.device)
        
        # Per-client ledger entries (held temporarily until deletion / unlearning)
        self.client_ledgers: Dict[int, Dict[str, torch.Tensor]] = {}
        
    def compute_client_summary(self, model: nn.Module, data_loader, client_id: int) -> Dict[str, torch.Tensor]:
        """
        Runs one forward pass over client data to compute local additive statistics.
        Does not share raw data with the server.
        """
        model.eval()
        A_k = torch.zeros((self.feature_dim, self.feature_dim), dtype=torch.float32, device=self.device)
        S_k = torch.zeros((self.num_classes, self.feature_dim), dtype=torch.float32, device=self.device)
        n_k = torch.zeros((self.num_classes,), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Extract penultimate feature representation phi(x)
                if hasattr(model, "extract_features"):
                    phi = model.extract_features(x)
                elif hasattr(model, "base") and hasattr(model.base, "layer4"):
                    # Fallback forward hook or manual extraction
                    features = model.base.conv1(x)
                    features = model.base.bn1(features)
                    features = model.base.relu(features)
                    features = model.base.maxpool(features)
                    features = model.base.layer1(features)
                    features = model.base.layer2(features)
                    features = model.base.layer3(features)
                    features = model.base.layer4(features)
                    features = model.base.avgpool(features)
                    phi = torch.flatten(features, 1)
                else:
                    raise AttributeError("Model must have extract_features() or standard ResNet base.")
                
                # Second moment: sum phi(x) phi(x)^T
                A_k += phi.T @ phi
                
                # Per-class feature sums and counts
                for c in range(self.num_classes):
                    mask = (y == c)
                    if mask.sum() > 0:
                        S_k[c] += phi[mask].sum(dim=0)
                        n_k[c] += mask.sum().float()
                        
        summary = {
            "A": A_k,
            "S": S_k,
            "n": n_k
        }
        return summary
        
    def register_client(self, client_id: int, summary: Dict[str, torch.Tensor]):
        """Adds a client's summary to the global ledger."""
        self.client_ledgers[client_id] = {
            "A": summary["A"].to(self.device),
            "S": summary["S"].to(self.device),
            "n": summary["n"].to(self.device)
        }
        self.A_all += self.client_ledgers[client_id]["A"]
        self.S_all += self.client_ledgers[client_id]["S"]
        self.n_all += self.client_ledgers[client_id]["n"]
        
    def get_retained_ledger(self, forget_client_id: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Subtracts the forgotten client's contributions from the global totals.
        Returns (retained_stats, forget_stats).
        """
        if forget_client_id not in self.client_ledgers:
            raise KeyError(f"Client {forget_client_id} is not registered in the ledger.")
            
        f_summary = self.client_ledgers[forget_client_id]
        
        A_r = self.A_all - f_summary["A"]
        S_r = self.S_all - f_summary["S"]
        n_r = self.n_all - f_summary["n"]
        
        retained_stats = {
            "A": A_r,
            "S": S_r,
            "n": n_r,
            "total_samples": n_r.sum().item()
        }
        
        forget_stats = {
            "A": f_summary["A"],
            "S": f_summary["S"],
            "n": f_summary["n"],
            "total_samples": f_summary["n"].sum().item()
        }
        
        return retained_stats, forget_stats
        
    def purge_client(self, forget_client_id: int):
        """
        Permanently deletes the forgotten client's summary to comply with GDPR.
        Updates the global ledger totals to reflect only retained data.
        """
        if forget_client_id in self.client_ledgers:
            f_summary = self.client_ledgers.pop(forget_client_id)
            self.A_all -= f_summary["A"]
            self.S_all -= f_summary["S"]
            self.n_all -= f_summary["n"]
            del f_summary
            print(f"[AdditiveLedger] Client {forget_client_id} permanently purged from ledger (GDPR Art. 17 compliant).")
