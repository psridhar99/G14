"""
train.py — training loop, evaluation loop, checkpointing, and experiment runner.
"""

import copy
import os
import pickle
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    DEVICE, EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    CHECKPOINT_DIR, LOG_DIR,
)


# ── Core loops ────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(DEVICE):
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = x.size(0)
        tot       += bs
        tot_loss  += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()

    return tot_loss / tot, tot_correct / tot


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)
        bs = x.size(0)
        tot       += bs
        tot_loss  += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()

    return tot_loss / tot, tot_correct / tot


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight_decay: float = WEIGHT_DECAY,   # ← now correctly passed through
    epochs: int = EPOCHS,
    checkpoint_path: str = None,
) -> Dict:
    """
    Train a model for `epochs` epochs, saving the best checkpoint.
    Resumes from `checkpoint_path` if it exists.

    Returns a dict with: exp_name, best_val_acc, history, best_state.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, exp_name))
    model    = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(DEVICE)

    history      = {"train_loss": [], "val_loss": [], "train_accs": [], "val_accs": []}
    best_val_acc = 0.0
    best_state   = None
    start_epoch  = 0

    # ── Resume from checkpoint if available ──────────────────────────────────
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        history      = ckpt["history"]
        best_val_acc = ckpt["best_val_acc"]
        best_state   = ckpt["best_state"]
        start_epoch  = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {start_epoch} "
              f"(best val acc so far: {best_val_acc:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())

        # Save checkpoint every epoch (enables resume on interruption)
        if checkpoint_path:
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict(),
                "history":         history,
                "best_val_acc":    best_val_acc,
                "best_state":      best_state,
            }, checkpoint_path)

    writer.close()
    return {
        "exp_name":     exp_name,
        "best_val_acc": best_val_acc,
        "history":      history,
        "best_state":   best_state,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_results(results: List[Dict], path: str) -> None:
    """Pickle a list of experiment result dicts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")


def load_results(path: str) -> List[Dict]:
    """Load pickled experiment result dicts."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load best_state weights from a checkpoint file into a model."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    key  = "best_state" if "best_state" in ckpt else "model_state"
    model.load_state_dict(ckpt[key])
    model.to(DEVICE)
    model.eval()
    return model
