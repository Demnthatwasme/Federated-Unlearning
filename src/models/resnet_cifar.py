import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights

def freeze_bn(model: nn.Module):
    """
    Freeze BatchNorm layers in eval mode to prevent running statistics
    from corrupting on single-class batches during extreme non-IID FL.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

class FedVisionModel(nn.Module):
    """
    Federated Vision Model adapted for extreme 1-class-per-client CIFAR-100 FL.
    Uses a ResNet-18 convolutional backbone with cosine-similarity normalized classifier
    and frozen BatchNorm running statistics to prevent client drift and representation collapse.
    """
    def __init__(
        self,
        num_classes: int = 10,
        scale: float = 20.0,
        img_size: int = 128,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.img_size = img_size
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base = models.resnet18(weights=weights)
        self.base.fc = nn.Identity()
        
        self.classifier = nn.Linear(512, num_classes, bias=False)
        nn.init.orthogonal_(self.classifier.weight)
        
        # Ensure BatchNorm is frozen from creation
        freeze_bn(self.base)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Convolutional target layer for Grad-CAM visual attribution."""
        return self.base.layer4

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim features from the ResNet-18 backbone."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )
        return self.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing scaled cosine logits."""
        feats = self.extract_features(x)
        f_norm = F.normalize(feats, dim=1)
        w_norm = F.normalize(self.classifier.weight, dim=1)
        logits = self.scale * F.linear(f_norm, w_norm)
        return logits

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return parameters updated during FL (layer4 + classifier)."""
        params = []
        for p in self.base.layer4.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.classifier.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def get_weights(self) -> List[np.ndarray]:
        """Convert PyTorch model state_dict to list of NumPy arrays for Flower."""
        return [v.detach().cpu().numpy() for _, v in self.state_dict().items()]

    def set_weights(self, weights: List[np.ndarray]):
        """Load list of NumPy arrays into PyTorch model state_dict."""
        state_dict = {}
        for (k, v), w in zip(self.state_dict().items(), weights):
            state_dict[k] = torch.as_tensor(w, dtype=v.dtype, device=v.device)
        self.load_state_dict(state_dict, strict=True)


def get_cifar_model(
    num_classes: int = 10,
    model_type: str = "fedvision",
    device: str = "cpu"
) -> FedVisionModel:
    """Factory function to instantiate the federated vision model."""
    model = FedVisionModel(num_classes=num_classes)
    return model.to(device)
