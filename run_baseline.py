"""
scripts/run_baseline.py — train the ResNet18 baseline on CIFAR-100.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --epochs 100
    python scripts/run_baseline.py --epochs 50 --lr 0.01

Resumes automatically from checkpoints/baseline_best.pt if it exists.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from config import BASELINE_CKPT, EPOCHS, SEED
from data import build_loaders
from models import build_resnet18
from train import run_experiment, save_results


def main(epochs: int = EPOCHS, lr: float = None):
    torch.manual_seed(SEED)
    print(f"\n{'='*60}\n  Baseline ResNet18 — CIFAR-100\n{'='*60}")

    train_loader, val_loader = build_loaders("standard")
    model = build_resnet18(pretrained=True)

    # Optionally override LR from CLI
    if lr is not None:
        from config import LR
        import config
        config.LR = lr

    result = run_experiment(
        exp_name        = "baseline/resnet18",
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        weight_decay    = 5e-4,
        epochs          = epochs,
        checkpoint_path = BASELINE_CKPT,
    )

    print(f"\nBest validation accuracy: {result['best_val_acc']:.4f}")
    print(f"Checkpoint saved to: {BASELINE_CKPT}")

    # Save result dict alongside checkpoint for notebook loading
    save_results([result], BASELINE_CKPT.replace(".pt", "_result.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr",     type=float, default=None)
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr)
