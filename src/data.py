"""
data.py — CIFAR-100 data loaders and augmentation strategies.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple

from config import (
    SEED, DATA_ROOT, TRAIN_VAL_SPLIT, BATCH_SIZE,
    CIFAR100_MEAN, CIFAR100_STD,
)


# ── Augmentation strategies ───────────────────────────────────────────────────

def get_transform(strategy: str = "standard") -> transforms.Compose:
    """
    Return a training transform for the given augmentation strategy.
    Strategies: 'none', 'standard', 'strong', 'cutout'
    """
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if strategy == "none":
        return transforms.Compose([transforms.ToTensor(), normalize])

    elif strategy == "standard":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    elif strategy == "strong":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ])

    elif strategy == "cutout":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=1.0, scale=(0.0625, 0.25),
                                     ratio=(1.0, 1.0)),
        ])

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy!r}")


def get_val_transform() -> transforms.Compose:
    """Deterministic validation transform (no augmentation)."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # matches paper preprocessing
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


# ── TTA transforms ────────────────────────────────────────────────────────────

def get_tta_transforms():
    """Return the list of test-time augmentation transforms."""
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    return [
        transforms.Compose([transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2),
                            transforms.ToTensor(), normalize]),
    ]


# ── Loader factory ────────────────────────────────────────────────────────────

def build_loaders(
    aug_strategy: str = "standard",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders from CIFAR-100 with an 80/20 split.
    Validation set always uses the deterministic val transform.
    """
    train_ds_full = datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True,
        transform=get_transform(aug_strategy),
    )
    val_ds_full = datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True,
        transform=get_val_transform(),
    )

    n_train = int(TRAIN_VAL_SPLIT * len(train_ds_full))
    n_val   = len(train_ds_full) - n_train
    gen     = torch.Generator().manual_seed(SEED)

    train_idx, val_idx = random_split(
        range(len(train_ds_full)), [n_train, n_val], generator=gen
    )

    # Use Subset wrappers so each split gets its own transform
    from torch.utils.data import Subset
    train_data = Subset(train_ds_full, list(train_idx))
    val_data   = Subset(val_ds_full,   list(val_idx))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


def get_raw_val_dataset_and_indices():
    """
    Return (raw_dataset_no_transform, val_indices) for TTA evaluation.
    raw_dataset has transform=None so TTA can apply its own transforms.
    """
    raw_ds = datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True, transform=None
    )
    n_train = int(TRAIN_VAL_SPLIT * len(raw_ds))
    gen     = torch.Generator().manual_seed(SEED)
    _, val_idx = random_split(
        range(len(raw_ds)), [n_train, len(raw_ds) - n_train], generator=gen
    )
    return raw_ds, list(val_idx)
