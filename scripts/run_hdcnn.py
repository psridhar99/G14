"""
scripts/run_hdcnn.py — build and train the HD-CNN.

Usage:
    python scripts/run_hdcnn.py                     # uses predefined CIFAR-100 superclasses
    python scripts/run_hdcnn.py --hierarchy learned  # derives groups from confusion matrix
    python scripts/run_hdcnn.py --epochs 50

Requires a trained baseline checkpoint at checkpoints/baseline_best.pt
(run run_baseline.py first).

Resumes automatically from checkpoints/hdcnn_best.pt if it exists.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torchvision import datasets
from config import (
    HDCNN_CKPT, BASELINE_CKPT, EPOCHS, SEED,
    DATA_ROOT, HDCNN_K, HDCNN_LAMBDA,
)
from data import build_loaders, get_val_transform
from torch.utils.data import DataLoader
from models import HDCNN, HDCNNLoss, build_resnet18, init_hdcnn_from_baseline
from train import load_model_weights, save_results
from ablation import build_confusion_matrix, build_hierarchy_from_confusion


# CIFAR-100 canonical superclass groupings (20 groups × 5 classes each)
CIFAR100_SUPERCLASSES = [
    [4, 30, 55, 72, 95],   # aquatic mammals
    [1, 32, 67, 73, 91],   # fish
    [54, 62, 70, 82, 92],  # flowers
    [9, 10, 16, 28, 61],   # food containers
    [0, 51, 53, 57, 83],   # fruit and vegetables
    [22, 39, 40, 86, 87],  # household electrical devices
    [5, 20, 25, 84, 94],   # household furniture
    [6, 7, 14, 18, 24],    # insects
    [3, 42, 43, 88, 97],   # large carnivores
    [12, 17, 37, 68, 76],  # large man-made outdoor things
    [23, 33, 49, 60, 71],  # large natural outdoor scenes
    [15, 19, 21, 31, 38],  # large omnivores and herbivores
    [34, 63, 64, 66, 75],  # medium-sized mammals
    [26, 45, 77, 79, 99],  # non-insect invertebrates
    [2, 11, 35, 46, 98],   # people
    [27, 29, 44, 78, 93],  # reptiles
    [36, 50, 65, 74, 80],  # small mammals
    [47, 52, 56, 59, 96],  # trees
    [8, 13, 48, 58, 90],   # vehicles 1
    [41, 69, 81, 85, 89],  # vehicles 2
]


def main(hierarchy: str = "cifar", epochs: int = EPOCHS):
    torch.manual_seed(SEED)
    print(f"\n{'='*60}\n  HD-CNN — CIFAR-100\n{'='*60}")

    train_loader, val_loader = build_loaders("standard")

    # ── Determine coarse groups ───────────────────────────────────────────────
    if hierarchy == "cifar":
        print("  Using CIFAR-100 predefined superclass hierarchy (20 groups)")
        coarse_groups = CIFAR100_SUPERCLASSES

    elif hierarchy == "learned":
        print("  Deriving hierarchy from confusion matrix of baseline model...")
        if not os.path.exists(BASELINE_CKPT):
            raise FileNotFoundError(
                f"Baseline checkpoint not found at {BASELINE_CKPT}. "
                "Run scripts/run_baseline.py first."
            )
        baseline = build_resnet18(pretrained=False)
        baseline = load_model_weights(baseline, BASELINE_CKPT)
        conf_mat = build_confusion_matrix(baseline, val_loader)
        coarse_groups = build_hierarchy_from_confusion(conf_mat, K=HDCNN_K)
        for k, g in enumerate(coarse_groups):
            print(f"  Coarse group {k:2d}: {len(g):3d} classes")
    else:
        raise ValueError(f"Unknown hierarchy option: {hierarchy!r}")

    # ── Build HD-CNN ──────────────────────────────────────────────────────────
    from config import DEVICE
    hd_model  = HDCNN(coarse_groups, num_fine=100, pretrained_backbone=False).to(DEVICE)
    hd_loss   = HDCNNLoss(coarse_groups, lam=HDCNN_LAMBDA)

    # Initialize from baseline if available and no HD-CNN checkpoint yet
    if not os.path.exists(HDCNN_CKPT) and os.path.exists(BASELINE_CKPT):
        print("  Initializing HD-CNN from baseline weights...")
        baseline = build_resnet18(pretrained=False)
        baseline = load_model_weights(baseline, BASELINE_CKPT)
        init_hdcnn_from_baseline(hd_model, baseline)
    elif not os.path.exists(BASELINE_CKPT):
        print("  Warning: no baseline checkpoint found. HD-CNN will train from pretrained ImageNet weights.")

    # ── Custom training loop for HD-CNN (uses HDCNNLoss, not CrossEntropy) ────
    import torch.optim as optim
    from config import LR, MOMENTUM, WEIGHT_DECAY, LOG_DIR
    from torch.utils.tensorboard import SummaryWriter
    import copy

    os.makedirs(os.path.dirname(HDCNN_CKPT), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    optimizer = optim.SGD(hd_model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(DEVICE)
    writer    = SummaryWriter(log_dir=os.path.join(LOG_DIR, "hdcnn"))

    history      = {"train_loss": [], "val_loss": [], "train_accs": [], "val_accs": []}
    best_val_acc = 0.0
    best_state   = None
    start_epoch  = 0

    # Resume if checkpoint exists
    if os.path.exists(HDCNN_CKPT):
        ckpt = torch.load(HDCNN_CKPT, map_location=DEVICE)
        hd_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history      = ckpt["history"]
        best_val_acc = ckpt["best_val_acc"]
        best_state   = ckpt["best_state"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {start_epoch} (best val acc: {best_val_acc:.4f})")

    for epoch in range(start_epoch, epochs):
        # ── Train ──
        hd_model.train()
        tot_loss, tot_correct, tot = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(DEVICE):
                final_probs, coarse_probs, _ = hd_model(x)
                loss, nll, consist = hd_loss(final_probs, coarse_probs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            tot       += bs
            tot_loss  += loss.item() * bs
            tot_correct += (final_probs.argmax(1) == y).sum().item()
        train_loss = tot_loss / tot
        train_acc  = tot_correct / tot

        # ── Validate ──
        hd_model.eval()
        vtot_loss, vtot_correct, vtot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                final_probs, coarse_probs, _ = hd_model(x)
                loss, _, _ = hd_loss(final_probs, coarse_probs, y)
                bs = x.size(0)
                vtot       += bs
                vtot_loss  += loss.item() * bs
                vtot_correct += (final_probs.argmax(1) == y).sum().item()
        val_loss = vtot_loss / vtot
        val_acc  = vtot_correct / vtot
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)
        print(f"  Epoch {epoch+1}/{epochs} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(hd_model.state_dict())

        torch.save({
            "epoch":           epoch,
            "model_state":     hd_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history":         history,
            "best_val_acc":    best_val_acc,
            "best_state":      best_state,
            "coarse_groups":   coarse_groups,
        }, HDCNN_CKPT)

    writer.close()
    print(f"\nBest HD-CNN validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {HDCNN_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", type=str, default="cifar",
                        choices=["cifar", "learned"],
                        help="'cifar' = use predefined superclasses; "
                             "'learned' = derive from confusion matrix")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(hierarchy=args.hierarchy, epochs=args.epochs)
