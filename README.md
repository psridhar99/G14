# G14 — Hierarchical Deep CNN for CIFAR-100 Classification

**DS 6050 | University of Virginia School of Data Science**  
Maneesh Malpeddi · Brooke Milligan · Thomas Schindler · Pranav Sridhar

---

## Overview

This project investigates hierarchical deep learning approaches for fine-grained image classification on CIFAR-100. It trains a ResNet-18 baseline, runs structured ablation studies across architecture variants and regularization strategies, evaluates test-time augmentation (TTA) inference strategies, and implements a full Hierarchical Deep CNN (HD-CNN) with data-driven coarse group discovery via spectral clustering.

---

## Requirements

### Environment

- Python 3.10–3.13 (Python 3.14 not supported by PyTorch)
- CUDA-capable GPU strongly recommended (tested on Google Colab T4/A100/H100)

> **Note for Colab:** GPU must be enabled manually. Go to **Runtime → Change runtime type → T4 GPU** (or better) before running.

> **Note for Windows:** Set `num_workers=0` and `pin_memory=False` in the DataLoader cells if you encounter hanging on startup.

---

## How to Run

Open `G14.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

CIFAR-100 will download automatically (~170 MB) on first run into a `./data/` directory.

The full notebook takes approximately:
- **~2–3 hours** on a Colab T4
- **~45–60 minutes** on an H100

---

## Notebook Structure & Expected Output

### Cell 1 — Imports
No output. Loads PyTorch, torchvision, sklearn, matplotlib, and utilities.

---

### Cell 3 — Configuration
```
Using device: cuda
```
If you see `cpu` here, your GPU is not enabled. Stop and fix this before proceeding — CPU training will take many hours.

---

### Cell 6 — Data Loading
No output. Downloads CIFAR-100 if not already cached, creates an 80/20 train/val split with a fixed seed (42), applies separate transforms to each split:

- **Train:** RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize
- **Val:** ToTensor + Normalize only (no augmentation)

---

### Cell 12 — Baseline ResNet-18 Training (100 epochs)
Prints one line per epoch. Expect val accuracy to climb steadily (sample values below):

```
Epoch 1/100  | Train: 0.301 | Val: 0.414
Epoch 10/100 | Train: 0.820 | Val: 0.663
Epoch 50/100 | Train: 0.997 | Val: 0.760
Epoch 100/100| Train: 1.000 | Val: 0.789
```

**Expected final result:** ~78–80% val accuracy. Train accuracy saturates to 1.000 around epoch 55–60, indicating overfitting, but val accuracy continues to slowly improve through 100 epochs. Model is saved to `resnet18_cifar100.pth`.

---

### Cell 13 — Baseline Training Curves
Renders a 2-panel matplotlib figure: loss curves (train vs val) and accuracy curves (train vs val). Expect a clear gap between train and val accuracy from epoch 50 onward — this is expected overfitting behavior.

---

### Cell 28 — Architecture Ablations
Trains 5 model variants for 100 epochs each. This is the longest cell. Example leaderboard at the end:

```
============================================================
  Architecture Ablations
============================================================
  Experiment                          Best Val Acc
  ----------------------------------- ------------
  arch/resnet50                             0.8220
  arch/resnet34                             0.7999
  arch/resnet18_baseline                    0.7795
  arch/shallow_resnet                       0.7604
  arch/resnet18_no_pretrain                 0.7529
```

ResNet-50 should be the clear winner. ResNet-18 without pretraining should be the weakest pretrained-architecture variant.

---

### Cell 29 — Regularization Ablations
Trains 4 regularization configurations for 100 epochs each. Example leaderboard:

```
============================================================
  Regularisation Ablations
============================================================
  Experiment                          Best Val Acc
  ----------------------------------- ------------
  reg/wd_aug                                0.7794
  reg/cutout_aug                            0.7722
  reg/aug_dropout                           0.7623
  reg/strong_aug_only                       0.7494
```

Weight decay + standard augmentation should be the top regularization config. Dropout should be the weakest performer.

---

### Cell 31 — Best Architecture Selection
```
Using best arch model: arch/resnet50 (val_acc=0.8220)
```

---

### Cell 32 — TTA Evaluation
Runs 10 TTA strategies on the best model (ResNet-50). Expected output:

```
============================================================
  Inference-Time Aggregation Results
============================================================
  Strategy                         Accuracy
  ------------------------------ ----------
  prob_avg_n3                        0.8400
  logit_avg_n5                       0.8390
  prob_avg_n5                        0.8380
  ...
  single_view                        0.8175
```

Probability averaging at n=3 should be the best strategy. All multi-view strategies should outperform single-view by 1–2+ points.

---

### Cell 36 — HD-CNN Hierarchy Construction
Builds 9 coarse groups from the baseline model's confusion matrix using spectral clustering. Expected output:

```
Coarse group 0: 4 classes
Coarse group 1: 57 classes
Coarse group 2: 1 classes
...
Coarse group 8: 24 classes
```

> **Note:** We found that the group sizes were highly imbalanced. This is expected — the spectral clustering reflects how the baseline model actually confuses classes, not the nominal CIFAR-100 superclass structure. Group 1 being very large (~57 classes) is a known limitation of the data-driven approach at K=9.

---

### Cell 37 — HD-CNN Weight Initialization
```
Weights initialized from baseline.
```

---

### Cell 41 — HD-CNN Training (4 × 100 epochs)
Trains the HD-CNN at four `w_coarse` values (0.1, 0.3, 0.5, 1.0), each for 100 epochs. Per-epoch output includes fine Top-1, Top-5, and coarse accuracy. At the end of each sweep:

```
Running w_coarse = 0.1
  Best (single-view) -> Top1: 0.763, Top5: 0.916, Coarse: 0.857
  Best (TTA)         -> Top1: 0.761, Top5: 0.921

Running w_coarse = 0.3
  Best (single-view) -> Top1: 0.758, Top5: 0.912, Coarse: 0.852
  ...
```

**Expected behavior:**
- Fine Top-1 will land in the **74–77% range** across all w_coarse values — below the baseline ResNet-18's 79%. This is expected given the imbalanced hierarchy.
- Coarse accuracy should reach **84–87%**, which is the HD-CNN's primary advantage over the flat baseline.
- All four w_coarse settings should perform within ~0.5% of each other on fine accuracy, showing low sensitivity to this hyperparameter.
- Lower w_coarse (0.1) tends to perform slightly better on fine Top-1.

---

### Cell 42 — Sample TTA Summary
```
w_coarse=0.1 | Single-view Top-1: 0.763 | TTA Top-1: 0.761
w_coarse=0.3 | Single-view Top-1: 0.758 | TTA Top-1: 0.760
w_coarse=0.5 | Single-view Top-1: 0.760 | TTA Top-1: 0.762
w_coarse=1.0 | Single-view Top-1: 0.759 | TTA Top-1: 0.755
```

TTA does not reliably improve HD-CNN performance — slight regressions are normal and expected due to augmentation noise in the probabilistic routing layer.

---

### Cell 47 — Final Results Table
Sample comparison across all models:

```
Model                                    Top-1   Top-5    Fine   Super   Params
arch/resnet50                           82.20%  96.47%       —       —   23.71M
arch/resnet34                           79.99%  95.22%       —       —   21.33M
arch/resnet18_baseline                  77.95%  94.40%       —       —   11.22M
...
HD-CNN (w_coarse=0.1)                   76.30%  91.84%  76.30%  85.84%  105.72M
```

---

### Cells 49–52 — Baseline Correlation Matrix
Renders a 100×100 heatmap of inter-class output correlations from the baseline model. Expected: strong diagonal (self-correlation = 1.0), near-zero off-diagonal values, with a few moderate positive correlations (~0.06–0.11) between visually similar classes.

---

### Cells 54–57 — HD-CNN Correlation Matrix
Same analysis on the HD-CNN. Expected: slightly more structured correlation pattern than the baseline, with higher max off-diagonal correlations (~0.65–0.70) reflecting the model's learned coarse-group structure.

---


## Reproducibility

All experiments use a fixed random seed (`SEED = 42`) set at the top of the notebook. Results may vary slightly across hardware due to non-deterministic CUDA operations, but should not fluctuate a lot.

---

## GitHub

[https://github.com/psridhar99/G14](https://github.com/psridhar99/G14)