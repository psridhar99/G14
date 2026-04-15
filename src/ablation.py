"""
ablation.py — architecture, regularization, and TTA ablation runners.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader

from config import (
    ARCH_VARIANTS, REGULARIZATION_CONFIGS,
    TTA_STRATEGIES, TTA_N_AUGMENTS, TTA_EVAL_SAMPLES,
    DEVICE, SEED, CHECKPOINT_DIR,
    EPOCHS,
)
from data import build_loaders, get_raw_val_dataset_and_indices, get_tta_transforms, get_val_transform
from models import build_arch_variant, build_reg_model
from train import run_experiment
import os


# ── Architecture ablations ────────────────────────────────────────────────────

def run_architecture_ablations(epochs: int = EPOCHS) -> List[Dict]:
    train_loader, val_loader = build_loaders("standard")
    results = []

    for variant in ARCH_VARIANTS:
        print(f"\n{'='*60}\n  ARCH ABLATION: {variant}\n{'='*60}")
        model = build_arch_variant(variant)
        ckpt  = os.path.join(CHECKPOINT_DIR, f"arch_{variant}.pt")
        res   = run_experiment(
            exp_name      = f"arch/{variant}",
            model         = model,
            train_loader  = train_loader,
            val_loader    = val_loader,
            weight_decay  = 5e-4,
            epochs        = epochs,
            checkpoint_path = ckpt,
        )
        results.append(res)

    return results


# ── Regularization ablations ──────────────────────────────────────────────────

def run_regularization_ablations(epochs: int = EPOCHS) -> List[Dict]:
    results = []

    for cfg in REGULARIZATION_CONFIGS:
        name    = cfg["name"]
        wd      = cfg["wd"]
        dropout = cfg["dropout"]
        aug     = cfg["aug"]

        print(f"\n{'='*60}\n  REG ABLATION: {name}\n{'='*60}")
        train_loader, val_loader = build_loaders(aug)
        model = build_reg_model(dropout=dropout)
        ckpt  = os.path.join(CHECKPOINT_DIR, f"reg_{name}.pt")

        res = run_experiment(
            exp_name      = f"reg/{name}",
            model         = model,
            train_loader  = train_loader,
            val_loader    = val_loader,
            weight_decay  = wd,      # ← correctly threaded through now
            epochs        = epochs,
            checkpoint_path = ckpt,
        )
        res["config"] = cfg
        results.append(res)

    return results


# ── TTA ablations ─────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_evaluate(
    model: nn.Module,
    raw_dataset,
    indices: List[int],
    strategy: str = "prob_avg",
    n_augments: int = 5,
) -> float:
    model.eval()
    tta_tfms = get_tta_transforms()[:n_augments]
    val_tfm  = get_val_transform()
    correct, total = 0, 0

    for idx in indices:
        raw_img, label = raw_dataset[idx]

        if strategy == "single":
            img  = val_tfm(raw_img).unsqueeze(0).to(DEVICE)
            pred = model(img).argmax(1).item()

        elif strategy in ("prob_avg", "logit_avg"):
            preds = []
            for tfm in tta_tfms:
                img = tfm(raw_img).unsqueeze(0).to(DEVICE)
                out = model(img)
                preds.append(torch.softmax(out, 1) if strategy == "prob_avg" else out)
            pred = torch.stack(preds).mean(0).argmax(1).item()

        elif strategy == "majority":
            votes = []
            for tfm in tta_tfms:
                img = tfm(raw_img).unsqueeze(0).to(DEVICE)
                votes.append(model(img).argmax(1).item())
            pred = max(set(votes), key=votes.count)

        else:
            raise ValueError(f"Unknown TTA strategy: {strategy!r}")

        correct += int(pred == label)
        total   += 1

    return correct / total


def run_tta_ablations(
    trained_model: nn.Module,
    n_eval_samples: int = TTA_EVAL_SAMPLES,
) -> Dict[str, float]:
    raw_ds, val_indices = get_raw_val_dataset_and_indices()
    indices = val_indices[:n_eval_samples]

    tta_results = {}
    for n_aug in TTA_N_AUGMENTS:
        for strat in TTA_STRATEGIES:
            if strat == "single" and n_aug > 1:
                continue
            key = "single_view" if strat == "single" else f"{strat}_n{n_aug}"
            print(f"  TTA: {strat!r}  n_aug={n_aug} ...", end=" ", flush=True)
            acc = tta_evaluate(trained_model, raw_ds, indices,
                               strategy=strat, n_augments=n_aug)
            tta_results[key] = acc
            print(f"acc={acc:.4f}")

    return tta_results


# ── Confusion-matrix hierarchy (for HD-CNN coarse groups) ────────────────────

@torch.no_grad()
def build_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 100,
) -> np.ndarray:
    model.eval()
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for x, y in loader:
        x = x.to(DEVICE)
        preds = model(x).argmax(dim=1).cpu().numpy()
        for t, p in zip(y.numpy(), preds):
            counts[t, p] += 1
    row_sums = counts.sum(axis=1, keepdims=True).clip(min=1)
    return counts / row_sums


def build_hierarchy_from_confusion(
    conf_mat: np.ndarray,
    K: int = 20,
    gamma: float = 5.0,
) -> List[List[int]]:
    """
    Derive overlapping coarse groups from a confusion matrix via
    spectral clustering.  Default K=20 matches CIFAR-100 superclasses.
    """
    D = 1.0 - conf_mat
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    affinity = np.exp(-D)

    sc     = SpectralClustering(n_clusters=K, affinity="precomputed",
                                random_state=SEED, n_init=10)
    labels = sc.fit_predict(affinity)

    disjoint = [[] for _ in range(K)]
    for cls, k in enumerate(labels):
        disjoint[k].append(cls)

    u_t        = 1.0 / (gamma * K)
    overlapping = [list(g) for g in disjoint]
    for k in range(K):
        in_group = set(disjoint[k])
        for j in range(conf_mat.shape[0]):
            if j in in_group:
                continue
            if conf_mat[j, list(in_group)].sum() >= u_t:
                overlapping[k].append(j)

    return [sorted(set(g)) for g in overlapping]


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def print_results_table(results: List[Dict], title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"  {'Experiment':<40} {'Best Val Acc':>12}")
    print(f"  {'-'*40} {'-'*12}")
    for r in sorted(results, key=lambda r: r["best_val_acc"], reverse=True):
        print(f"  {r['exp_name']:<40} {r['best_val_acc']:>12.4f}")
    print()


def print_tta_table(tta_results: Dict[str, float]) -> None:
    print(f"\n{'='*60}\n  Inference-Time Aggregation Results\n{'='*60}")
    print(f"  {'Strategy':<30} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for k, v in sorted(tta_results.items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {v:>10.4f}")
    print()
