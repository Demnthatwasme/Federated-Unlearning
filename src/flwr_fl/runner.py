import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from src.data.cifar_loader import SingleClassFLDataManager
from src.models.resnet_cifar import FedVisionModel, get_cifar_model, freeze_bn
from src.flwr_fl.client import CIFARFlowerClient

def evaluate_model_on_loader(
    model: nn.Module,
    data_loader,
    device: str = "cuda"
) -> Tuple[float, float]:
    """Evaluates average cross-entropy loss and top-1 accuracy on any DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += len(labels)
            
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def eval_client_split(
    model: nn.Module,
    test_loader,
    forget_client_id: int,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates retain vs forget accuracy and prediction confidence.
    """
    model.eval()
    f_correct, f_total = 0, 0
    r_correct, r_total = 0, 0
    f_conf_sum = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            
            f_mask = (labels == forget_client_id)
            r_mask = (labels != forget_client_id)
            
            if f_mask.any():
                f_correct += (preds[f_mask] == labels[f_mask]).sum().item()
                f_total += f_mask.sum().item()
                # Average probability assigned to the true forget class
                f_conf_sum += probs[f_mask, forget_client_id].sum().item()
                
            if r_mask.any():
                r_correct += (preds[r_mask] == labels[r_mask]).sum().item()
                r_total += r_mask.sum().item()
                
    overall_acc = (f_correct + r_correct) / max(1, (f_total + r_total))
    retain_acc = r_correct / max(1, r_total)
    forget_acc = f_correct / max(1, f_total)
    forget_conf = f_conf_sum / max(1, f_total)
    
    return {
        "overall_acc": float(overall_acc),
        "retain_acc": float(retain_acc),
        "forget_acc": float(forget_acc),
        "forget_conf": float(forget_conf)
    }

def eval_multiclass_client_split(
    model: nn.Module,
    test_loader,
    forget_train_loader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates multi-class federated unlearning metrics:
      - overall_acc / retain_acc: accuracy on holdout CIFAR-100 test set
      - test_loss: cross-entropy loss on holdout test set
      - forget_train_acc: accuracy on the forgotten client's private training data (D_forget)
      - forget_train_loss: cross-entropy loss on D_forget (higher indicates unlearning of memorized private data)
      - forget_train_conf: mean softmax confidence on the true labels of D_forget
    """
    model.eval()
    t_loss, t_acc = evaluate_model_on_loader(model, test_loader, device=device)
    f_loss, f_acc = evaluate_model_on_loader(model, forget_train_loader, device=device)
    
    f_conf_sum = 0.0
    f_total = 0
    with torch.no_grad():
        for images, labels in forget_train_loader:
            images, labels = images.to(device), labels.to(device)
            probs = F.softmax(model(images), dim=1)
            for i in range(len(labels)):
                f_conf_sum += probs[i, labels[i]].item()
                f_total += 1
    f_conf = f_conf_sum / max(1, f_total)
    
    return {
        "overall_acc": float(t_acc),
        "test_loss": float(t_loss),
        "retain_acc": float(t_acc),
        "forget_acc": float(f_acc),
        "forget_train_acc": float(f_acc),
        "forget_train_loss": float(f_loss),
        "forget_train_conf": float(f_conf),
        "forget_conf": float(f_conf)
    }

def initialize_model_prototypes(
    model: FedVisionModel,
    data_manager: Any,
    client_ids: List[int],
    device: str = "cuda"
):
    """
    Initializes each class prototype vector in the classifier head
    using the mean normalized feature representations of training data.
    Works for both SingleClassFLDataManager (1 class/client) and MultiClassFLDataManager.
    """
    freeze_bn(model)
    class_feats = {c: [] for c in range(data_manager.num_classes)}
    with torch.no_grad():
        for cid in client_ids:
            loader = data_manager.get_client_train_loader(cid)
            for x, y in loader:
                x = x.to(device)
                f = model.extract_features(x)
                for i in range(len(y)):
                    label = y[i].item()
                    if label in class_feats:
                        class_feats[label].append(f[i:i+1])
    for c in range(data_manager.num_classes):
        if class_feats[c]:
            avg_f = F.normalize(torch.cat(class_feats[c]).mean(dim=0), dim=0)
            with torch.no_grad():
                model.classifier.weight[c] = avg_f

def run_fl_training(
    data_manager: SingleClassFLDataManager,
    client_ids: Optional[List[int]] = None,
    num_rounds: int = 5,
    local_epochs: int = 1,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_checkpoint_path: Optional[str] = None,
    forget_client_id: Optional[int] = None,
    tag: str = "FL",
    verbose: bool = True
) -> Tuple[FedVisionModel, Dict[str, Any], List[Dict[str, torch.Tensor]], List[Dict[int, Dict[str, torch.Tensor]]]]:
    """
    Executes Federated Learning using Flower NumPyClient instances.
    Logs historical global models and client updates per round for FedEraser unlearning.
    """
    if client_ids is None:
        client_ids = list(range(data_manager.num_classes))
    if forget_client_id is None:
        forget_client_id = data_manager.num_classes - 1

    # Initialize Global Model
    global_model = get_cifar_model(num_classes=data_manager.num_classes, device=device)
    initialize_model_prototypes(global_model, data_manager, client_ids, device=device)
    
    # Store initial model checkpoint
    history_global_models = [copy.deepcopy(global_model.state_dict())]
    history_client_updates = [] # round-by-round client updates: [ {cid: {param_name: delta_tensor}} ]

    # Initialize Flower Clients
    fl_clients = {}
    for cid in client_ids:
        client_train_loader = data_manager.get_client_train_loader(cid)
        client_val_loader = data_manager.get_forget_test_loader(cid)
        client_model = get_cifar_model(num_classes=data_manager.num_classes, device=device)
        
        fl_clients[cid] = CIFARFlowerClient(
            client_id=cid,
            model=client_model,
            train_loader=client_train_loader,
            val_loader=client_val_loader,
            device=device,
            learning_rate=learning_rate
        )

    full_test_loader = data_manager.get_full_test_loader()
    is_multiclass = hasattr(data_manager, "client_indices")

    def do_eval(m):
        if is_multiclass:
            return eval_multiclass_client_split(
                m, full_test_loader, data_manager.get_forget_train_loader(forget_client_id), device=device
            )
        else:
            return eval_client_split(m, full_test_loader, forget_client_id, device=device)
    
    # Initial evaluation
    initial_metrics = do_eval(global_model)
    target_name = f"Client {forget_client_id}" if is_multiclass else f"Client {forget_client_id} ('{data_manager.class_names[forget_client_id]}')"
    if verbose:
        print(f"\n[{tag}] Starting FL Training ({len(client_ids)} Clients, {num_rounds} Rounds)")
        print(f"[{tag}] Clients: {client_ids} | Forget Target: {target_name}")
        forget_tag = "Forget Train Acc" if is_multiclass else "Forget Acc"
        print(f"[{tag}] Round 00 Init | Retain/Test Acc: {initial_metrics['retain_acc']:.2%} | {forget_tag}: {initial_metrics['forget_acc']:.2%} | Forget Conf: {initial_metrics['forget_conf']:.3f}")

    history = {
        "round": [0],
        "train_loss": [0.0],
        "train_acc": [0.0],
        "overall_acc": [initial_metrics["overall_acc"]],
        "retain_acc": [initial_metrics["retain_acc"]],
        "forget_acc": [initial_metrics["forget_acc"]],
        "forget_conf": [initial_metrics["forget_conf"]],
    }

    t0 = time.time()
    for r in range(1, num_rounds + 1):
        r_start = time.time()
        g_state = copy.deepcopy(global_model.state_dict())
        global_weights = global_model.get_weights()
        
        fit_results = []
        round_client_updates = {}
        total_samples = 0
        fit_config = {
            "local_epochs": str(local_epochs),
            "learning_rate": str(learning_rate),
            "is_calibration": "False"
        }

        # Local training on each Flower client
        for cid in client_ids:
            client = fl_clients[cid]
            updated_weights, num_samples, metrics = client.fit(global_weights, fit_config)
            fit_results.append((cid, updated_weights, num_samples, metrics))
            total_samples += num_samples
            
            # Record client update delta = client_weights - global_weights (as tensors)
            client_state = client.model.state_dict()
            delta = {}
            for k in g_state:
                if g_state[k].is_floating_point() and any(t in k for t in ["layer4", "classifier"]):
                    delta[k] = (client_state[k] - g_state[k]).detach().clone()
            round_client_updates[cid] = delta

        history_client_updates.append(round_client_updates)

        # Server FedAvg Parameter Aggregation
        new_state = copy.deepcopy(g_state)
        first_cid = client_ids[0]
        for k in round_client_updates[first_cid]:
            weighted_deltas = [
                (num_samples / max(1, total_samples)) * round_client_updates[cid][k].float()
                for cid, _, num_samples, _ in fit_results
            ]
            new_state[k] = g_state[k] + torch.stack(weighted_deltas).sum(dim=0).to(g_state[k].dtype)
            
        global_model.load_state_dict(new_state)
        history_global_models.append(copy.deepcopy(global_model.state_dict()))

        # Evaluate round performance
        eval_metrics = do_eval(global_model)
        train_loss = sum(m["train_loss"] * n for _, _, n, m in fit_results) / max(1, total_samples)
        train_acc = sum(m["train_acc"] * n for _, _, n, m in fit_results) / max(1, total_samples)

        history["round"].append(r)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["overall_acc"].append(float(eval_metrics["overall_acc"]))
        history["retain_acc"].append(float(eval_metrics["retain_acc"]))
        history["forget_acc"].append(float(eval_metrics["forget_acc"]))
        history["forget_conf"].append(float(eval_metrics["forget_conf"]))

        r_time = time.time() - r_start
        if verbose:
            print(
                f"[{tag}] Round {r:02d}/{num_rounds:02d} | "
                f"Retain Acc: {eval_metrics['retain_acc']:.2%} | "
                f"Forget Acc: {eval_metrics['forget_acc']:.2%} | "
                f"Forget Conf: {eval_metrics['forget_conf']:.3f} | "
                f"Overall: {eval_metrics['overall_acc']:.2%} | "
                f"Time: {r_time:.2f}s"
            )

    total_time = time.time() - t0
    history["total_time"] = float(total_time)

    if save_checkpoint_path:
        os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
        torch.save(global_model.state_dict(), save_checkpoint_path)
        if verbose:
            print(f"[{tag}] Saved final model checkpoint to {save_checkpoint_path}")

    return global_model, history, history_global_models, history_client_updates
