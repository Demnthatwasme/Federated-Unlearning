import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from src.models.resnet_cifar import freeze_bn

class CIFARFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing an individual federated client.
    In the single-class setting, each client holds training samples for 1 specific CIFAR-100 class.
    Supports standard FL training rounds as well as FedEraser calibration rounds.
    """
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return current local model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def set_parameters(self, parameters: List[np.ndarray]):
        """Update local model weights with received parameters."""
        self.model.set_weights(parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train local model using received parameters and local single-class data.
        Supports standard multi-epoch training and fast FedEraser calibration steps.
        """
        self.set_parameters(parameters)
        freeze_bn(self.model)

        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("learning_rate", self.learning_rate))
        is_calibration = bool(config.get("is_calibration", False))
        max_steps = int(config.get("cali_steps", 999999))

        # Train only layer4 and classifier head
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.base.layer4.parameters():
            p.requires_grad = True
        for p in self.model.classifier.parameters():
            p.requires_grad = True

        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.model.train()
        freeze_bn(self.model)
        total_loss = 0.0
        correct = 0
        total_samples = 0
        steps = 0

        stop_training = False
        for _ in range(epochs):
            if stop_training:
                break
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += len(labels)
                steps += 1

                if is_calibration and steps >= max_steps:
                    stop_training = True
                    break

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "client_id": self.client_id,
            "train_loss": float(avg_loss),
            "train_acc": float(acc),
            "steps": steps
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate local model on holdout / validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total_samples += len(labels)

        avg_loss = total_loss / max(1, total_samples)
        acc = correct / max(1, total_samples)

        return float(avg_loss), total_samples, {
            "client_id": self.client_id,
            "val_loss": float(avg_loss),
            "val_acc": float(acc)
        }
