import time
import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np

from src.data.cifar_loader import SingleClassFLDataManager
from src.models.resnet_cifar import get_cifar_model, ResNet18CIFAR
from src.flwr_fl.runner import evaluate_model_on_loader, run_fl_training
from src.flwr_fl.client import CIFARFlowerClient

class FederatedUnlearningEngine:
    """
    Engine implementing Federated Unlearning methods:
      1. Exact Retraining (Gold Standard Baseline M_retrain)
      2. Fast Federated Gradient Ascent + Repair Calibration (M_unlearn)
    """
    def __init__(
        self,
        data_manager: SingleClassFLDataManager,
        forget_client_id: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data_manager = data_manager
        self.forget_client_id = forget_client_id
        self.device = device
        self.retain_client_ids = [
            cid for cid in range(data_manager.num_classes) if cid != forget_client_id
        ]
        
    def run_exact_retrain(
        self,
        num_rounds: int = 15,
        local_epochs: int = 2,
        learning_rate: float = 0.02,
        save_path: Optional[str] = "checkpoints/M_retrain.pt",
        verbose: bool = True
    ) -> Tuple[ResNet18CIFAR, Dict]:
        """
        Gold standard: completely retrain FL model using ONLY retain clients.
        """
        start_time = time.time()
        if verbose:
            print(f"\n==========================================")
            print(f"=== EXACT RETRAINING BASELINE (GOLD STANDARD) ===")
            print(f"==========================================")
            print(f"Excluding Client {self.forget_client_id} ({self.data_manager.class_names[self.forget_client_id]})")
            print(f"Retaining Clients: {self.retain_client_ids}")

        model, history = run_fl_training(
            data_manager=self.data_manager,
            client_ids=self.retain_client_ids,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.device,
            save_checkpoint_path=save_path,
            forget_client_id=self.forget_client_id,
            verbose=verbose
        )
        elapsed = time.time() - start_time
        metrics = self.evaluate_all(model)
        metrics["runtime_seconds"] = elapsed
        metrics["method"] = "Exact Retraining"
        if verbose:
            print(f"Retraining completed in {elapsed:.2f}s | Retain Test Acc: {metrics['retain_test_acc']:.3f} | Forget Test Acc: {metrics['forget_test_acc']:.3f}")
        return model, metrics

    def run_federated_ga_ft(
        self,
        original_model: ResNet18CIFAR,
        ga_epochs: int = 3,
        ga_lr: float = 0.005,
        repair_rounds: int = 3,
        repair_epochs: int = 1,
        repair_lr: float = 0.008,
        save_path: Optional[str] = "checkpoints/M_unlearn.pt",
        verbose: bool = True
    ) -> Tuple[ResNet18CIFAR, Dict]:
        """
        Fast Federated Unlearning via:
          Phase 1: Scrubbing - Gradient Ascent on the forget client's data (D_forget) with gradient clipping
          Phase 2: Repair - Rapid Federated Averaging fine-tuning across retain clients (D_retain) using Flower clients
        """
        start_time = time.time()
        unlearned_model = copy.deepcopy(original_model).to(self.device)
        
        if verbose:
            print(f"\n==========================================")
            print(f"=== FAST FEDERATED UNLEARNING (GA-FT) ===")
            print(f"==========================================")
            print(f"Unlearning Client {self.forget_client_id} ({self.data_manager.class_names[self.forget_client_id]})")
            print(f"Phase 1: Gradient Ascent Scrubbing ({ga_epochs} epochs, lr={ga_lr})")

        # ---------------- Phase 1: Gradient Ascent Scrubbing ----------------
        forget_loader = self.data_manager.get_forget_train_loader(self.forget_client_id, shuffle=True)
        unlearned_model.train()
        criterion = nn.CrossEntropyLoss()
        
        # We perform gradient ascent by taking positive gradient steps (maximizing loss)
        ga_optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=ga_lr, momentum=0.9)
        
        for ep in range(ga_epochs):
            total_loss = 0.0
            for images, labels in forget_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                ga_optimizer.zero_grad()
                outputs = unlearned_model(images)
                loss = criterion(outputs, labels)
                
                # Invert loss to maximize CrossEntropy (gradient ascent)
                neg_loss = -loss
                neg_loss.backward()
                
                # Clip gradients to prevent numeric explosion
                torch.nn.utils.clip_grad_norm_(unlearned_model.parameters(), max_norm=1.0)
                ga_optimizer.step()
                total_loss += loss.item()
                
            if verbose:
                avg_l = total_loss / max(1, len(forget_loader))
                _, f_acc = evaluate_model_on_loader(unlearned_model, forget_loader, device=self.device)
                print(f"  [GA Epoch {ep+1}/{ga_epochs}] Forget Loss: {avg_l:.4f} | Forget Train Acc: {f_acc:.3f}")

        # ---------------- Phase 2: Federated Retain Calibration via Flower ----------------
        if verbose:
            print(f"Phase 2: Federated Repair Calibration ({repair_rounds} rounds across {len(self.retain_client_ids)} retain clients)")

        global_weights = unlearned_model.get_weights()
        
        # Setup Flower clients for retain clients
        fl_clients = {}
        for cid in self.retain_client_ids:
            client_model = get_cifar_model(num_classes=self.data_manager.num_classes, device=self.device)
            fl_clients[cid] = CIFARFlowerClient(
                client_id=cid,
                model=client_model,
                train_loader=self.data_manager.get_client_train_loader(cid),
                val_loader=self.data_manager.get_forget_test_loader(cid),
                device=self.device,
                learning_rate=repair_lr
            )

        for r in range(1, repair_rounds + 1):
            fit_results = []
            total_samples = 0
            config = {"local_epochs": str(repair_epochs), "learning_rate": str(repair_lr)}

            for cid in self.retain_client_ids:
                client = fl_clients[cid]
                updated_w, n_k, metrics = client.fit(global_weights, config)
                fit_results.append((updated_w, n_k, metrics))
                total_samples += n_k

            # FedAvg aggregation
            new_global_weights = []
            for i, w in enumerate(global_weights):
                if np.issubdtype(w.dtype, np.floating):
                    agg_w = np.zeros_like(w, dtype=np.float64)
                    for client_w, n_k, _ in fit_results:
                        weight_factor = n_k / max(1, total_samples)
                        agg_w += weight_factor * client_w[i]
                    new_global_weights.append(agg_w.astype(w.dtype))
                else:
                    new_global_weights.append(fit_results[0][0][i].copy())
            global_weights = new_global_weights
            unlearned_model.set_weights(global_weights)

            if verbose:
                _, ret_acc = evaluate_model_on_loader(
                    unlearned_model,
                    self.data_manager.get_retain_test_loader(self.forget_client_id),
                    device=self.device
                )
                _, f_acc = evaluate_model_on_loader(
                    unlearned_model,
                    self.data_manager.get_forget_test_loader(self.forget_client_id),
                    device=self.device
                )
                print(f"  [Repair Round {r}/{repair_rounds}] Retain Test Acc: {ret_acc:.3f} | Forget Test Acc: {f_acc:.3f}")

        elapsed = time.time() - start_time
        metrics = self.evaluate_all(unlearned_model)
        metrics["runtime_seconds"] = elapsed
        metrics["method"] = "Federated GA-FT"

        if save_path:
            torch.save(unlearned_model.state_dict(), save_path)
            if verbose:
                print(f"Saved unlearned model checkpoint to {save_path}")

        if verbose:
            print(f"Fast Unlearning completed in {elapsed:.2f}s | Retain Test Acc: {metrics['retain_test_acc']:.3f} | Forget Test Acc: {metrics['forget_test_acc']:.3f}")

        return unlearned_model, metrics

    def evaluate_all(self, model: nn.Module) -> Dict[str, float]:
        """Compute evaluation metrics across all relevant datasets."""
        full_test_loader = self.data_manager.get_full_test_loader()
        retain_test_loader = self.data_manager.get_retain_test_loader(self.forget_client_id)
        forget_test_loader = self.data_manager.get_forget_test_loader(self.forget_client_id)
        forget_train_loader = self.data_manager.get_forget_train_loader(self.forget_client_id)
        retain_train_loader = self.data_manager.get_retain_train_loader(self.forget_client_id)

        _, overall_test_acc = evaluate_model_on_loader(model, full_test_loader, self.device)
        _, retain_test_acc = evaluate_model_on_loader(model, retain_test_loader, self.device)
        _, forget_test_acc = evaluate_model_on_loader(model, forget_test_loader, self.device)
        _, forget_train_acc = evaluate_model_on_loader(model, forget_train_loader, self.device)
        _, retain_train_acc = evaluate_model_on_loader(model, retain_train_loader, self.device)

        return {
            "overall_test_acc": float(overall_test_acc),
            "retain_test_acc": float(retain_test_acc),
            "forget_test_acc": float(forget_test_acc),
            "forget_train_acc": float(forget_train_acc),
            "retain_train_acc": float(retain_train_acc),
        }
