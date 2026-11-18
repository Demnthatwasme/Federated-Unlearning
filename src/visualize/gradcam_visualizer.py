import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

class GradCAMExplainer:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    on PyTorch convolutional vision models.
    Uses input gradient autograd to guarantee robust gradient flow.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = "cuda"):
        self.model = model
        self.target_layer = target_layer
        self.device = device

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Generate Grad-CAM heatmap for a single image tensor (1, C, H, W) or (C, H, W).
        
        Returns:
            cam_heatmap: (H, W) float array normalized to [0, 1]
            predicted_class: int
            confidence: float (softmax probability of predicted class)
            all_probs: (num_classes,) array
        """
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        x = input_tensor.clone().detach().to(self.device).requires_grad_(True)
        
        activations = {}
        def hook_fn(_, inp, outp):
            activations["act"] = outp
            
        handle = self.target_layer.register_forward_hook(hook_fn)
        
        with torch.enable_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            pred_class = int(probs.argmax().item())
            confidence = float(probs[pred_class].item())
            all_probs = probs.detach().cpu().numpy()
            
            if target_class is None:
                target_class = pred_class
                
            score = logits[0, target_class]
            grads = torch.autograd.grad(score, activations["act"])[0]
            
        handle.remove()
        
        # Channel-wise global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)
        act = activations["act"]
        cam = F.relu((weights * act).sum(dim=1, keepdim=True)).squeeze()
        
        # Interpolate to match original input size (e.g., 32x32)
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam_resized = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        c_min, c_max = cam_resized.min(), cam_resized.max()
        if c_max - c_min > 1e-8:
            cam_norm = (cam_resized - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_resized)
            
        return cam_norm, pred_class, confidence, all_probs

    @staticmethod
    def overlay_heatmap(
        img_rgb: np.ndarray,
        cam_heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Overlays a Grad-CAM heatmap onto an RGB image normalized in [0, 1]."""
        heatmap_u8 = np.uint8(255 * np.clip(cam_heatmap, 0.0, 1.0))
        heatmap_color = cv2.applyColorMap(heatmap_u8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
        
        overlay = alpha * heatmap_rgb + (1.0 - alpha) * img_rgb
        return np.clip(overlay, 0.0, 1.0)


def compute_cam_drift(cam1: np.ndarray, cam2: np.ndarray) -> float:
    """
    Computes attribution drift between two CAM heatmaps:
    Drift = 1 - Pearson Correlation.
    0.0 = identical attention, 1.0 = uncorrelated attention.
    """
    a = cam1.flatten() - cam1.mean()
    b = cam2.flatten() - cam2.mean()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    corr = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(1.0 - corr, 0.0, 2.0))


def generate_unlearning_gallery(
    models_dict: Dict[str, nn.Module],
    sample_images: torch.Tensor,
    sample_labels: List[int],
    class_names: List[str],
    unnormalizer,
    forget_client_id: int,
    save_path: str = "results/gradcam_unlearning_comparison.png",
    device: str = "cuda",
    is_forget_sample: Optional[List[bool]] = None
) -> Dict[str, Any]:
    """
    Generates a publication-quality side-by-side visual gallery comparing:
      1. Original Input Image
      2. M_orig Grad-CAM
      3. M_unlearn (FedEraser) Grad-CAM
      4. M_retrain (Gold Standard) Grad-CAM
      5. Attribution Difference Heatmap |CAM_orig - CAM_unlearn|
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_samples = len(sample_images)
    
    # Setup explainers
    explainers = {
        name: GradCAMExplainer(m, m.gradcam_target_layer, device=device)
        for name, m in models_dict.items()
    }

    cols = ["Input Image", "Original FL (M_orig)", "FedEraser (M_unlearn)", "Retrained (M_retrain)", "Attribution Shift (|ΔCAM|)"]
    fig, axes = plt.subplots(n_samples, len(cols), figsize=(3.2 * len(cols), 3.0 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    drift_records = []

    for i in range(n_samples):
        img_t = sample_images[i]
        true_label = sample_labels[i]
        true_name = class_names[true_label]
        if is_forget_sample is not None and i < len(is_forget_sample):
            is_forget = is_forget_sample[i]
        else:
            is_forget = (true_label == forget_client_id)
        
        img_rgb = unnormalizer(img_t)
        
        # 1. Input Image
        axes[i, 0].imshow(img_rgb)
        type_str = "FORGOTTEN CLIENT" if is_forget else "RETAIN CLIENT"
        color_str = "red" if is_forget else "green"
        axes[i, 0].set_ylabel(f"[{type_str}]\nClass: {true_name}", fontsize=10, fontweight="bold", color=color_str)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        cams = {}
        confs = {}
        preds = {}
        
        for col_idx, (model_name, expl) in enumerate(explainers.items(), start=1):
            # Generate CAM for the model's actual predicted class to visualize what it attends to
            cam, pred_c, conf, _ = expl.generate_cam(img_t, target_class=None)
            overlay = expl.overlay_heatmap(img_rgb, cam, alpha=0.5)
            axes[i, col_idx].imshow(overlay)
            axes[i, col_idx].set_xticks([])
            axes[i, col_idx].set_yticks([])
            
            cams[model_name] = cam
            confs[model_name] = conf
            preds[model_name] = pred_c
            
            pred_name = class_names[pred_c]
            p_color = "black" if pred_c == true_label else "darkred"
            axes[i, col_idx].set_title(
                f"Pred: {pred_name} ({conf:.1%})",
                fontsize=9,
                color=p_color,
                fontweight="bold" if pred_c == true_label else "normal"
            )

        # 5. Difference Heatmap |CAM_orig - CAM_unlearn|
        cam_orig = cams.get("Original", list(cams.values())[0])
        cam_unl = cams.get("FedEraser", list(cams.values())[1])
        diff_cam = np.abs(cam_orig - cam_unl)
        diff_norm = (diff_cam - diff_cam.min()) / (diff_cam.max() - diff_cam.min() + 1e-8)
        diff_overlay = expl.overlay_heatmap(img_rgb, diff_norm, alpha=0.6, colormap=cv2.COLORMAP_MAGMA)
        
        drift = compute_cam_drift(cam_orig, cam_unl)
        drift_records.append({
            "sample_idx": i,
            "true_class": true_name,
            "is_forget": is_forget,
            "cam_drift": drift
        })

        axes[i, 4].imshow(diff_overlay)
        axes[i, 4].set_xticks([])
        axes[i, 4].set_yticks([])
        axes[i, 4].set_title(f"CAM Drift: {drift:.3f}", fontsize=9, fontweight="bold")

    # Column titles
    for c, title in enumerate(cols):
        axes[0, c].set_title(f"{title}\n" + axes[0, c].get_title(), fontsize=10, fontweight="bold")

    fig.suptitle(
        "Federated Visual Unlearning Evidence (Grad-CAM Attention Comparison)\n"
        f"Right to be Forgotten Request: Client {forget_client_id} ('{class_names[forget_client_id]}')",
        fontsize=12,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    
    return {
        "saved_path": save_path,
        "drift_records": drift_records,
        "avg_forget_cam_drift": float(np.mean([r["cam_drift"] for r in drift_records if r["is_forget"]])),
        "avg_retain_cam_drift": float(np.mean([r["cam_drift"] for r in drift_records if not r["is_forget"]]))
    }
