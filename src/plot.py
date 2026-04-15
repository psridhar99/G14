"""
plot.py — visualization utilities for training curves and ablation results.
All functions save to FIGURES_DIR and optionally display inline.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_history(history: Dict, title: str = "Training Curves",
                 save_name: str = None, show: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_accs"], label="Train")
    axes[1].plot(history["val_accs"],   label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()

    if save_name:
        path = os.path.join(FIGURES_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_ablation_bar(results: List[Dict], title: str = "Ablation Results",
                      save_name: str = None, show: bool = True) -> None:
    names = [r["exp_name"].split("/")[-1] for r in results]
    accs  = [r["best_val_acc"] for r in results]
    order = np.argsort(accs)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh([names[i] for i in order], [accs[i] for i in order])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel("Best Validation Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, max(accs) + 0.05)
    plt.tight_layout()

    if save_name:
        path = os.path.join(FIGURES_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_history(results: List[Dict], title: str = "Validation Accuracy Comparison",
                       save_name: str = None, show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        label = r["exp_name"].split("/")[-1]
        ax.plot(r["history"]["val_accs"], label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if save_name:
        path = os.path.join(FIGURES_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close(fig)
