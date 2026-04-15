"""
scripts/run_ablation.py — run architecture, regularization, and TTA ablations.

Usage:
    python scripts/run_ablation.py                    # all three
    python scripts/run_ablation.py --study arch
    python scripts/run_ablation.py --study reg
    python scripts/run_ablation.py --study tta        # requires baseline ckpt

Results are saved to checkpoints/ablation_results.pkl and can be loaded
in notebooks/02_ablation.ipynb without re-training.

Each individual experiment also saves its own .pt checkpoint, so you can
interrupt and resume at any point.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from config import (
    ABLATION_RESULTS, BASELINE_CKPT, EPOCHS, SEED,
    ARCH_VARIANTS,
)
from ablation import (
    run_architecture_ablations,
    run_regularization_ablations,
    run_tta_ablations,
    print_results_table,
    print_tta_table,
)
from models import build_arch_variant
from train import save_results, load_results, load_model_weights


def main(study: str = "all", epochs: int = EPOCHS):
    torch.manual_seed(SEED)
    all_results = {}

    # ── Architecture ablations ────────────────────────────────────────────────
    if study in ("all", "arch"):
        print(f"\n{'='*60}\n  Running architecture ablations\n{'='*60}")
        arch_results = run_architecture_ablations(epochs=epochs)
        all_results["arch"] = arch_results
        print_results_table(arch_results, "Architecture Ablations")

    # ── Regularization ablations ──────────────────────────────────────────────
    if study in ("all", "reg"):
        print(f"\n{'='*60}\n  Running regularization ablations\n{'='*60}")
        reg_results = run_regularization_ablations(epochs=epochs)
        all_results["reg"] = reg_results
        print_results_table(reg_results, "Regularization Ablations")

    # ── TTA ablations (needs a trained model) ─────────────────────────────────
    if study in ("all", "tta"):
        print(f"\n{'='*60}\n  Running TTA ablations\n{'='*60}")

        # Find the best arch model to run TTA on
        if "arch" in all_results:
            best = max(all_results["arch"], key=lambda r: r["best_val_acc"])
            variant = best["exp_name"].replace("arch/", "")
            tta_model = build_arch_variant(variant)
            tta_model.load_state_dict(best["best_state"])
        elif os.path.exists(BASELINE_CKPT):
            print(f"  Loading baseline from {BASELINE_CKPT}")
            from models import build_resnet18
            tta_model = build_resnet18(pretrained=False)
            tta_model = load_model_weights(tta_model, BASELINE_CKPT)
        else:
            print("  No trained model found. Run --study arch or train baseline first.")
            return

        tta_results = run_tta_ablations(tta_model)
        all_results["tta"] = tta_results
        print_tta_table(tta_results)

    # ── Save everything ───────────────────────────────────────────────────────
    save_results(all_results, ABLATION_RESULTS)
    print(f"\nAll ablation results saved to {ABLATION_RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",  type=str, default="all",
                        choices=["all", "arch", "reg", "tta"],
                        help="Which ablation study to run")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(study=args.study, epochs=args.epochs)
