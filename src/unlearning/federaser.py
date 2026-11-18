import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.data.cifar_loader import SingleClassFLDataManager, MultiClassFLDataManager
from src.models.resnet_cifar import FedVisionModel, get_cifar_model, freeze_bn
from src.flwr_fl.client import CIFARFlowerClient
from src.flwr_fl.runner import eval_client_split, eval_multiclass_client_split, run_fl_training

class FedEraserEngine:
    """
    Implements the FedEraser algorithm (Wang et al., IEEE/ACM IWQoS 2021)
    for federated client-level unlearning ("Right to be Forgotten").

    Instead of expensive retraining from scratch, FedEraser leverages historical
    updates stored at the central server and reconstructs the unlearned model
    along the training trajectory using fast calibration steps on retained clients.
    Works for both Single-Class (1 class/client) and Multi-Class (Dirichlet non-IID) settings.
    """
    def __init__(
        self,
        data_manager: Any,
        forget_client_id: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data_manager = data_manager
        self.forget_client_id = forget_client_id
        self.device = device
        self.num_classes = data_manager.num_classes
        self.is_multiclass = hasattr(data_manager, "client_indices")
        num_total_clients = getattr(data_manager, "num_clients", data_manager.num_classes)
        self.retain_client_ids = [
            cid for cid in range(num_total_clients) if cid != forget_client_id
        ]
        self.forget_class_name = data_manager.class_names[forget_client_id] if not self.is_multiclass else f"Client {forget_client_id}"

    def run_federaser(
        self,
        history_global_models: List[Dict[str, torch.Tensor]],
        history_client_updates: List[Dict[int, Dict[str, torch.Tensor]]],
        cali_epochs: int = 1,
        cali_steps: int = 4,
        cali_lr: float = 0.001,
        milestone_interval: int = 2,
        save_path: Optional[str] = "checkpoints/M_unlearn.pt",
        verbose: bool = True
    ) -> Tuple[FedVisionModel, Dict[str, Any], Dict[str, List[float]]]:
        """
        Executes FedEraser unlearning reconstruction:
          1. Start from initial global checkpoint W_0 with forget class weights neutralized.
          2. Select milestone calibration rounds (e.g. every milestone_interval rounds).
          3. For each milestone round:
             - Retained clients run fast calibration training from current unlearned state W'_{t-1}.
             - Server calibrates update direction using historical update magnitudes.
             - Server aggregates calibrated updates over retain clients only.
        """
        t0 = time.time()
        T = len(history_client_updates)
        milestone_rounds = [t for t in range(T) if (t + 1) % milestone_interval == 0 or t == T - 1]
        
        if verbose:
            print("\n" + "=" * 75)
            print("=== FEDERASER UNLEARNING RECONSTRUCTION ===")
            print("=" * 75)
            print(f"Target Forgotten Client: Client {self.forget_client_id} ('{self.forget_class_name}')")
            print(f"Retained Clients: {self.retain_client_ids}")
            print(f"Total FL Rounds: {T} | Milestone Cali Rounds: {[r+1 for r in milestone_rounds]} | Cali Steps/Client: {cali_steps} | LR: {cali_lr}")

        # Step 1: Initialize unlearning model from W_0
        unlearn_model = get_cifar_model(num_classes=self.num_classes, device=self.device)
        init_state = {k: v.clone().detach().to(self.device) for k, v in history_global_models[0].items()}
        
        # Ensure all history updates are on self.device
        device_updates = []
        for round_dict in history_client_updates:
            dev_r = {}
            for cid, delta_dict in round_dict.items():
                dev_r[cid] = {k: v.clone().detach().to(self.device) for k, v in delta_dict.items()}
            device_updates.append(dev_r)
        history_client_updates = device_updates

        # Neutralize the forgotten client's prototype in the classifier head only for single-class mode
        if not self.is_multiclass:
            with torch.no_grad():
                neutral_weight = torch.randn(512, device=self.device)
                neutral_weight = F.normalize(neutral_weight, dim=0)
                init_state["classifier.weight"][self.forget_client_id] = neutral_weight
            
        unlearn_model.load_state_dict(init_state)
        freeze_bn(unlearn_model)

        # Setup Flower clients for retained clients
        fl_clients = {}
        for cid in self.retain_client_ids:
            client_model = get_cifar_model(num_classes=self.num_classes, device=self.device)
            fl_clients[cid] = CIFARFlowerClient(
                client_id=cid,
                model=client_model,
                train_loader=self.data_manager.get_client_train_loader(cid),
                val_loader=self.data_manager.get_forget_test_loader(cid),
                device=self.device,
                learning_rate=cali_lr
            )

        full_test_loader = self.data_manager.get_full_test_loader()
        def do_eval(m):
            if self.is_multiclass:
                return eval_multiclass_client_split(
                    m, full_test_loader, self.data_manager.get_forget_train_loader(self.forget_client_id), device=self.device
                )
            else:
                return eval_client_split(m, full_test_loader, self.forget_client_id, device=self.device)

        init_eval = do_eval(unlearn_model)
        
        history = {
            "round": [0],
            "retain_acc": [init_eval["retain_acc"]],
            "forget_acc": [init_eval["forget_acc"]],
            "forget_conf": [init_eval["forget_conf"]],
            "overall_acc": [init_eval["overall_acc"]],
        }

        # Step 2: Trajectory Reconstruction over historical rounds
        cali_config = {
            "local_epochs": str(cali_epochs),
            "learning_rate": str(cali_lr),
            "is_calibration": "True",
            "cali_steps": str(cali_steps)
        }

        for t in range(T):
            r_start = time.time()
            curr_state = copy.deepcopy(unlearn_model.state_dict())
            curr_weights = unlearn_model.get_weights()
            new_state = copy.deepcopy(curr_state)
            first_cid = self.retain_client_ids[0]

            if t in milestone_rounds:
                # Milestone calibration round: retained clients calibrate update direction
                calibrated_updates = {}
                for cid in self.retain_client_ids:
                    client = fl_clients[cid]
                    updated_weights, num_samples, _ = client.fit(curr_weights, cali_config)
                    
                    loc_state = client.model.state_dict()
                    cali_delta = {}
                    for k in curr_state:
                        if curr_state[k].is_floating_point() and any(tag in k for tag in ["layer4", "classifier"]):
                            cali_delta[k] = (loc_state[k] - curr_state[k]).detach().clone()
                            
                    # Scale calibration update to preserve historical momentum magnitude
                    hist_delta = history_client_updates[t][cid]
                    calibrated_updates[cid] = {}
                    for k in cali_delta:
                        h_norm = hist_delta[k].norm() + 1e-8
                        c_norm = cali_delta[k].norm() + 1e-8
                        calibrated_updates[cid][k] = (cali_delta[k] / c_norm) * h_norm

                for k in calibrated_updates[first_cid]:
                    agg_delta = torch.stack([calibrated_updates[c][k].float() for c in self.retain_client_ids]).mean(dim=0)
                    new_state[k] = curr_state[k] + agg_delta.to(curr_state[k].dtype)
                cali_tag = "Milestone Cali"
            else:
                # Intermediate round: apply retained clients historical updates
                for k in history_client_updates[t][first_cid]:
                    agg_delta = torch.stack([history_client_updates[t][c][k].float() for c in self.retain_client_ids]).mean(dim=0)
                    new_state[k] = curr_state[k] + agg_delta.to(curr_state[k].dtype)
                cali_tag = "Fast Forward"

            # In single-class mode, keep forgotten class prototype neutralized
            if not self.is_multiclass:
                new_state["classifier.weight"][self.forget_client_id] = init_state["classifier.weight"][self.forget_client_id]
            unlearn_model.load_state_dict(new_state)

            # Evaluate round
            r_eval = do_eval(unlearn_model)
            history["round"].append(t + 1)
            history["retain_acc"].append(float(r_eval["retain_acc"]))
            history["forget_acc"].append(float(r_eval["forget_acc"]))
            history["forget_conf"].append(float(r_eval["forget_conf"]))
            history["overall_acc"].append(float(r_eval["overall_acc"]))

            r_time = time.time() - r_start
            if verbose:
                print(
                    f"  [FedEraser Round {t+1:02d}/{T:02d} ({cali_tag})] "
                    f"Retain Acc: {r_eval['retain_acc']:.2%} | "
                    f"Forget Acc: {r_eval['forget_acc']:.2%} | "
                    f"Forget Conf: {r_eval['forget_conf']:.3f} | "
                    f"Time: {r_time:.2f}s"
                )

        unlearn_time = time.time() - t0
        final_metrics = do_eval(unlearn_model)
        final_metrics["runtime_seconds"] = float(unlearn_time)
        final_metrics["method"] = "FedEraser"

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(unlearn_model.state_dict(), save_path)
            if verbose:
                print(f"Saved FedEraser model checkpoint to {save_path}")

        if verbose:
            print(f"FedEraser Completed in {unlearn_time:.2f}s | Retain Acc: {final_metrics['retain_acc']:.2%} | Forget Acc: {final_metrics['forget_acc']:.2%}")

        return unlearn_model, final_metrics, history

    def run_exact_retrain(
        self,
        num_rounds: int = 5,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        save_path: Optional[str] = "checkpoints/M_retrain.pt",
        verbose: bool = True
    ) -> Tuple[FedVisionModel, Dict[str, Any], Dict[str, List[float]]]:
        """
        Exact Retraining Gold Standard:
        Completely retrains the FL model from scratch using ONLY the retained clients
        for all rounds and full local epochs.
        """
        t0 = time.time()
        if verbose:
            print("\n" + "=" * 75)
            print("=== EXACT RETRAINING BASELINE (GOLD STANDARD) ===")
            print("=" * 75)
            print(f"Excluding Client {self.forget_client_id} ('{self.forget_class_name}')")
            print(f"Retained Clients: {self.retain_client_ids} | Rounds: {num_rounds} | Local Epochs: {local_epochs}")

        retrain_model, history, _, _ = run_fl_training(
            data_manager=self.data_manager,
            client_ids=self.retain_client_ids,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=self.device,
            save_checkpoint_path=save_path,
            forget_client_id=self.forget_client_id,
            tag="Retrain",
            verbose=verbose
        )

        total_time = history.get("total_time", time.time() - t0)
        full_test_loader = self.data_manager.get_full_test_loader()
        def do_eval(m):
            if self.is_multiclass:
                return eval_multiclass_client_split(
                    m, full_test_loader, self.data_manager.get_forget_train_loader(self.forget_client_id), device=self.device
                )
            else:
                return eval_client_split(m, full_test_loader, self.forget_client_id, device=self.device)

        final_metrics = do_eval(retrain_model)
        final_metrics["runtime_seconds"] = float(total_time)
        final_metrics["method"] = "Exact Retraining"

        if verbose:
            print(f"Exact Retraining Completed in {total_time:.2f}s | Retain Acc: {final_metrics['retain_acc']:.2%} | Forget Acc: {final_metrics['forget_acc']:.2%}")

        return retrain_model, final_metrics, history
