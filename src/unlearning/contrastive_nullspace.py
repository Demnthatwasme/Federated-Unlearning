"""
Fed-CNS: Contrastive Null-Space Scrubbing Engine for Federated Vision Unlearning.

Implements server-only contrastive null-space unlearning powered by the Additive Ledger:
1. Solves the generalized Rayleigh quotient:
     max_V tr(V^T Sigma_f V)  s.t.  V^T (Sigma_r + rho * I) V = I_k
   to find directions dominating the forgotten client's representations with minimal
   retained energy.
2. Builds the null-space scrubbing projector P = I - V_k V_k^T.
3. Refreshes class prototype centroids strictly from retained data sums:
     mu_c^r = S_{r, c} / n_{r, c}
     W_c = normalize(mu_c^r)
4. Modifies the classifier head in closed-form:
     W_unlearned = W_refreshed @ P

Runtime: < 50 milliseconds.
Retained Clients: 100% Offline (Zero Wake-Up).
Server Storage: O(d^2 + C * d) (~1 MB total).
"""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from src.unlearning.additive_ledger import AdditiveLedger

class ContrastiveNullSpaceEngine:
    def __init__(
        self,
        rho: float = 1e-4,
        energy_threshold: float = 0.95,
        max_k: int = 64,
        device: str = "cpu"
    ):
        self.rho = rho
        self.energy_threshold = energy_threshold
        self.max_k = max_k
        self.device = torch.device(device)
        
    def solve_generalized_eigenproblem(
        self,
        Sigma_f: torch.Tensor,
        Sigma_r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves Sigma_f @ v = lambda * (Sigma_r + rho * I) @ v.
        Returns eigenvalues (descending) and eigenvectors V.
        """
        d = Sigma_r.shape[0]
        # Regularized retained covariance
        Sigma_r_reg = Sigma_r + self.rho * torch.eye(d, device=self.device)
        
        # Cholesky factor: Sigma_r_reg = L @ L.T
        try:
            L = torch.linalg.cholesky(Sigma_r_reg)
        except torch.linalg.LinAlgError:
            # Fallback to stronger regularization if poorly conditioned
            Sigma_r_reg += 1e-3 * torch.eye(d, device=self.device)
            L = torch.linalg.cholesky(Sigma_r_reg)
            
        Linv = torch.linalg.inv(L)
        # Transform to standard symmetric eigenproblem: C = Linv @ Sigma_f @ Linv.T
        C = Linv @ Sigma_f @ Linv.T
        C = 0.5 * (C + C.T)  # Ensure exact symmetry
        
        evals, evecs = torch.linalg.eigh(C)
        
        # Sort in descending order
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # Transform back to generalized eigenvectors: V = Linv.T @ evecs
        V = Linv.T @ evecs
        # Normalize columns of V
        V = V / torch.norm(V, dim=0, keepdim=True)
        return evals, V

    def build_nullspace_projector(
        self,
        Sigma_f: torch.Tensor,
        Sigma_r: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Extracts the private forgotten subspace V_k and builds projector P = I - V_k @ V_k.T.
        """
        evals, V = self.solve_generalized_eigenproblem(Sigma_f, Sigma_r)
        
        if top_k is None:
            # Choose k based on cumulative forget eigenvalue energy
            pos_evals = torch.clamp(evals, min=0.0)
            cum_energy = torch.cumsum(pos_evals, dim=0) / (pos_evals.sum() + 1e-8)
            # Find index reaching energy_threshold
            k_candidates = (cum_energy >= self.energy_threshold).nonzero()
            if len(k_candidates) > 0:
                k = min(int(k_candidates[0].item()) + 1, self.max_k)
            else:
                k = min(16, self.max_k)
            k = max(1, k)
        else:
            k = min(top_k, self.max_k)
            
        V_k = V[:, :k]  # [d, k]
        
        # Orthonormalize V_k via QR for numerical stability
        Q, _ = torch.linalg.qr(V_k)
        
        # Orthogonal Null-space projector: P = I - Q @ Q.T
        d = Sigma_r.shape[0]
        I = torch.eye(d, device=self.device)
        P = I - Q @ Q.T
        
        return P, k, evals[:k]

    def unlearn(
        self,
        base_model: nn.Module,
        ledger: AdditiveLedger,
        forget_client_id: int,
        top_k: Optional[int] = None
    ) -> Tuple[nn.Module, Dict]:
        """
        Performs server-only unlearning in closed form without touching retained clients.
        """
        t0 = time.time()
        unlearned_model = copy.deepcopy(base_model).to(self.device)
        unlearned_model.eval()
        
        retained_stats, forget_stats = ledger.get_retained_ledger(forget_client_id)
        
        n_r = max(1.0, retained_stats["total_samples"])
        n_f = max(1.0, forget_stats["total_samples"])
        
        Sigma_r = retained_stats["A"] / n_r
        Sigma_f = forget_stats["A"] / n_f
        
        # Step 1: Build contrastive null-space projector
        P, k_used, top_evals = self.build_nullspace_projector(Sigma_f, Sigma_r, top_k=top_k)
        
        # Step 2: Refresh classifier prototypes from retained ledger
        # Head weights W in CosineClassifier: shape [num_classes, 512]
        with torch.no_grad():
            if hasattr(unlearned_model, "classifier"):
                head_linear = unlearned_model.classifier
            elif hasattr(unlearned_model, "fc"):
                head_linear = unlearned_model.fc
            else:
                raise AttributeError("Model has neither 'classifier' nor 'fc' attribute.")
                
            W_head = head_linear.weight.data.clone()  # [C, d]
            
            S_r = retained_stats["S"]  # [C, d]
            n_r_classes = retained_stats["n"]  # [C]
            
            for c in range(ledger.num_classes):
                cnt = n_r_classes[c].item()
                if cnt > 0:
                    # Re-center prototype on retained client feature centroid
                    mu_c = S_r[c] / cnt
                    mu_c_norm = mu_c / torch.norm(mu_c, p=2).clamp(min=1e-8)
                    W_head[c] = mu_c_norm
                else:
                    # Setting A: Entire class belonged to forget client!
                    # Neutralize prototype row so it never triggers argmax
                    W_head[c] = torch.zeros_like(W_head[c])
                    
            # Step 3: Project prototype vectors into the contrastive null space
            # W_scrubbed = W_head @ P
            W_scrubbed = W_head @ P
            
            # Re-normalize retained prototypes to preserve hypersphere geometry
            for c in range(ledger.num_classes):
                if n_r_classes[c].item() > 0:
                    norm = torch.norm(W_scrubbed[c], p=2).clamp(min=1e-8)
                    W_scrubbed[c] = W_scrubbed[c] / norm
                    
            head_linear.weight.data.copy_(W_scrubbed)
            
        runtime_ms = (time.time() - t0) * 1000.0
        
        metrics = {
            "unlearn_method": "Fed-CNS (Contrastive Null Space)",
            "runtime_ms": runtime_ms,
            "runtime_seconds": runtime_ms / 1000.0,
            "subspace_dimension_k": k_used,
            "top_eigenvalues": top_evals.detach().cpu().tolist(),
            "retained_samples": int(n_r),
            "forgotten_samples": int(n_f),
            "retained_clients_online": 0,
            "server_storage_mb": (ledger.feature_dim**2 * 4) / (1024 * 1024)
        }
        
        return unlearned_model, metrics
