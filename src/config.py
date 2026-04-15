"""
config.py — single source of truth for all hyperparameters, paths, and constants.
Edit here; everything else imports from here.
"""

import os
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 100        # paper claims 100; notebook had 50 — unified here
LR         = 0.01
MOMENTUM   = 0.9
WEIGHT_DECAY = 5e-4     # default; overridden per-experiment in ablation configs
BATCH_SIZE = 128

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_ROOT      = "./data"
TRAIN_VAL_SPLIT = 0.8
NUM_CLASSES    = 100
CIFAR100_MEAN  = (0.5071, 0.4867, 0.4408)
CIFAR100_STD   = (0.2675, 0.2565, 0.2761)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "runs"
FIGURES_DIR    = "figures"

BASELINE_CKPT     = os.path.join(CHECKPOINT_DIR, "baseline_best.pt")
ABLATION_RESULTS  = os.path.join(CHECKPOINT_DIR, "ablation_results.pkl")
HDCNN_CKPT        = os.path.join(CHECKPOINT_DIR, "hdcnn_best.pt")

# ── Ablation configs ──────────────────────────────────────────────────────────
ARCH_VARIANTS = [
    "resnet18_baseline",
    "resnet18_no_pretrain",
    "resnet34",
    "resnet50",
]

REGULARIZATION_CONFIGS = [
    {"name": "baseline",        "wd": 5e-4, "dropout": 0.0, "aug": "standard"},
    {"name": "wd_aug",          "wd": 5e-4, "dropout": 0.0, "aug": "standard"},
    {"name": "aug_dropout",     "wd": 0.0,  "dropout": 0.5, "aug": "strong"},
    {"name": "strong_aug_only", "wd": 0.0,  "dropout": 0.0, "aug": "strong"},
    {"name": "cutout_aug",      "wd": 5e-4, "dropout": 0.0, "aug": "cutout"},
    {"name": "dropout_only",    "wd": 0.0,  "dropout": 0.5, "aug": "none"},
    {"name": "wd_only",         "wd": 5e-4, "dropout": 0.0, "aug": "none"},
    {"name": "all_reg",         "wd": 5e-4, "dropout": 0.5, "aug": "strong"},
]

# ── TTA ───────────────────────────────────────────────────────────────────────
TTA_STRATEGIES  = ["single", "prob_avg", "logit_avg", "majority"]
TTA_N_AUGMENTS  = [1, 3, 5]
TTA_EVAL_SAMPLES = 2000

# ── HD-CNN ────────────────────────────────────────────────────────────────────
HDCNN_K         = 20    # use CIFAR-100's 20 superclasses by default
HDCNN_LAMBDA    = 20.0  # consistency loss weight
