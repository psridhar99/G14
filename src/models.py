"""
models.py — ResNet baseline variants and HD-CNN architecture.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, DEVICE


# ── ResNet baseline helpers ───────────────────────────────────────────────────

def _patch_for_cifar(model: nn.Module, out_features: int = NUM_CLASSES) -> nn.Module:
    """
    Adapt any ResNet for 32×32 CIFAR images:
      - 7×7 conv → 3×3 conv (retains spatial resolution)
      - maxpool → Identity (prevents resolution collapse)
      - final FC → 100-class head
    """
    in_ch = model.conv1.in_channels
    model.conv1  = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc     = nn.Linear(model.fc.in_features, out_features)
    return model


def build_resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    return _patch_for_cifar(models.resnet18(weights=weights))


def build_resnet34(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    return _patch_for_cifar(models.resnet34(weights=weights))


def build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    return _patch_for_cifar(models.resnet50(weights=weights))


def build_arch_variant(variant: str) -> nn.Module:
    """Factory for all architecture ablation variants."""
    if variant == "resnet18_baseline":
        return build_resnet18(pretrained=True)
    elif variant == "resnet18_no_pretrain":
        return build_resnet18(pretrained=False)
    elif variant == "resnet34":
        return build_resnet34(pretrained=True)
    elif variant == "resnet50":
        return build_resnet50(pretrained=True)
    else:
        raise ValueError(f"Unknown architecture variant: {variant!r}")


# ── Dropout variant ───────────────────────────────────────────────────────────

class ResNet18WithDropout(nn.Module):
    """ResNet18 with a dropout layer inserted before the final FC."""

    def __init__(self, dropout_p: float = 0.5, pretrained: bool = True):
        super().__init__()
        base = build_resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(512, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_reg_model(dropout: float = 0.0, pretrained: bool = True) -> nn.Module:
    """Return the appropriate model for a regularization config."""
    if dropout > 0.0:
        return ResNet18WithDropout(dropout_p=dropout, pretrained=pretrained)
    return build_resnet18(pretrained=pretrained)


# ── HD-CNN ────────────────────────────────────────────────────────────────────

class SharedBackbone(nn.Module):
    """ResNet18 stem through layer2 — shared by coarse and all fine components."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        bb = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # CIFAR patch on the shared stem
        bb.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bb.maxpool = nn.Identity()
        self.stem = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class CoarseComponent(nn.Module):
    """
    Coarse component: independent layer3/layer4/GAP/FC(→100) with
    fine-to-coarse aggregation for routing weights.
    """

    def __init__(self, coarse_groups, num_fine: int = NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()
        self.coarse_groups = coarse_groups
        self.K = len(coarse_groups)
        self.num_fine = num_fine

        bb = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backend = nn.Sequential(bb.layer3, bb.layer4,
                                     nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(512, num_fine)

        # coarse_mask[k, j] = 1 if fine class j belongs to coarse group k
        mask = torch.zeros(self.K, num_fine)
        for k, group in enumerate(coarse_groups):
            for j in group:
                mask[k, j] = 1.0
        self.register_buffer("coarse_mask", mask)

    def forward(self, shared_feat: torch.Tensor):
        feat        = self.backend(shared_feat).flatten(1)      # (B, 512)
        fine_logits = self.fc(feat)                              # (B, 100)
        fine_probs  = torch.softmax(fine_logits, dim=1)

        coarse_probs = (
            fine_probs.unsqueeze(1) * self.coarse_mask.unsqueeze(0)
        ).sum(dim=2)                                             # (B, K)
        coarse_probs = coarse_probs / coarse_probs.sum(
            dim=1, keepdim=True).clamp(min=1e-8)

        return fine_logits, fine_probs, coarse_probs


class FineComponent(nn.Module):
    """One fine-grained component: independent layer3/layer4/GAP/FC(→|group|)."""

    def __init__(self, group_size: int, pretrained: bool = False):
        super().__init__()
        bb = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.backend = nn.Sequential(bb.layer3, bb.layer4,
                                     nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(512, group_size)

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        feat = self.backend(shared_feat).flatten(1)
        return self.fc(feat)                                     # (B, |group|)


class HDCNN(nn.Module):
    """
    Full HD-CNN following Yan et al. 2015:
      shared backbone → coarse component + K fine components
                     → probabilistic averaging layer
    """

    def __init__(self, coarse_groups, num_fine: int = NUM_CLASSES,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.coarse_groups = coarse_groups
        self.K        = len(coarse_groups)
        self.num_fine = num_fine

        self.shared = SharedBackbone(pretrained=pretrained_backbone)
        self.coarse = CoarseComponent(coarse_groups, num_fine,
                                      pretrained=pretrained_backbone)
        self.fine_components = nn.ModuleList([
            FineComponent(len(g), pretrained=False) for g in coarse_groups
        ])
        self._group_tensors = [
            torch.tensor(g, dtype=torch.long) for g in coarse_groups
        ]

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        self._group_tensors = [t.to(device) for t in self._group_tensors]
        return self

    def forward(self, x: torch.Tensor, active_k=None):
        shared_feat = self.shared(x)
        fine_logits_B, _, coarse_probs = self.coarse(shared_feat)

        if active_k is None:
            active_k = list(range(self.K))

        final_probs = torch.zeros(x.size(0), self.num_fine,
                                  device=x.device)
        weight_sum  = torch.zeros(x.size(0), 1, device=x.device)

        for k in active_k:
            w_k      = coarse_probs[:, k].unsqueeze(1)
            logits_k = self.fine_components[k](shared_feat)
            probs_k  = torch.softmax(logits_k, dim=1)
            idx      = self._group_tensors[k]
            final_probs.scatter_add_(
                1,
                idx.unsqueeze(0).expand(x.size(0), -1),
                w_k * probs_k,
            )
            weight_sum += w_k

        final_probs = final_probs / weight_sum.clamp(min=1e-8)
        return final_probs, coarse_probs, fine_logits_B


class HDCNNLoss(nn.Module):
    """
    Combined loss: NLL(final_probs, targets) + λ * Σ_k (t_k - mean_batch(B_ik))²
    t_k = fraction of training images in coarse group k (uniform assumption).
    """

    def __init__(self, coarse_groups, num_fine: int = NUM_CLASSES,
                 lam: float = 20.0):
        super().__init__()
        self.lam = lam
        self.K   = len(coarse_groups)
        total    = sum(len(g) for g in coarse_groups)
        t        = torch.tensor([len(g) / total for g in coarse_groups])
        self.register_buffer("t_k", t)

    def forward(self, final_probs: torch.Tensor, coarse_probs: torch.Tensor,
                targets: torch.Tensor):
        nll = -torch.log(
            final_probs[range(len(targets)), targets].clamp(min=1e-8)
        ).mean()
        batch_mean  = coarse_probs.mean(dim=0)
        consistency = self.lam * ((self.t_k - batch_mean) ** 2).sum()
        return nll + consistency, nll.item(), consistency.item()


# ── Weight init from baseline ─────────────────────────────────────────────────

def init_hdcnn_from_baseline(hd_model: HDCNN, baseline: nn.Module) -> None:
    """Copy matching weights from a trained flat ResNet18 into an HD-CNN."""
    sd = baseline.state_dict()

    # Shared backbone (stem indices: conv1=0, bn1=1, relu=2, maxpool=3, layer1=4, layer2=5)
    name_to_idx = {"conv1": 0, "bn1": 1, "layer1": 4, "layer2": 5}
    shared_sd   = {}
    for name, idx in name_to_idx.items():
        for k, v in sd.items():
            if k.startswith(name + "."):
                shared_sd[f"stem.{idx}.{k[len(name)+1:]}"] = v
    hd_model.shared.load_state_dict(shared_sd, strict=False)

    # Coarse component (layer3=0, layer4=1 in backend Sequential)
    coarse_sd = {}
    for i, name in enumerate(["layer3", "layer4"]):
        for k, v in sd.items():
            if k.startswith(name + "."):
                coarse_sd[f"backend.{i}.{k[len(name)+1:]}"] = v
    for k, v in sd.items():
        if k.startswith("fc."):
            coarse_sd[f"fc.{k[3:]}"] = v
    hd_model.coarse.load_state_dict(coarse_sd, strict=False)

    # Seed all fine component backends from baseline layer3/layer4
    for fine_comp in hd_model.fine_components:
        fine_sd = {}
        for i, name in enumerate(["layer3", "layer4"]):
            for k, v in sd.items():
                if k.startswith(name + "."):
                    fine_sd[f"backend.{i}.{k[len(name)+1:]}"] = v
        fine_comp.load_state_dict(fine_sd, strict=False)

    print("HD-CNN weights initialized from baseline ResNet18.")
