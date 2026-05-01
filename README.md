# G14 — Multi-Label Wildlife Species Prediction with CNN and HD-CNN

## Project structure

```
G14/
├── src/                   # All reusable code — define once, import everywhere
│   ├── config.py          # Single source of truth: hyperparams, paths, seeds
│   ├── data.py            # CIFAR-100 loaders and augmentation strategies
│   ├── models.py          # ResNet variants, ResNet18WithDropout, full HD-CNN
│   ├── train.py           # Training loop, eval loop, checkpointing, run_experiment()
│   ├── ablation.py        # Arch / reg / TTA runners + confusion-matrix hierarchy
│   └── plot.py            # Curve and bar chart helpers (save to figures/)
│
├── scripts/               # Run once — train and save checkpoints
│   ├── run_baseline.py    # Train ResNet18 baseline
│   ├── run_ablation.py    # Run arch / reg / TTA ablations
│   └── run_hdcnn.py       # Train HD-CNN (resumes from baseline weights)
│
├── notebooks/             # Open after training — load checkpoints, visualize only
│   ├── 01_baseline.ipynb  # Curves, error analysis
│   ├── 02_ablation.ipynb  # Arch / reg / TTA comparison plots
│   └── 03_hdcnn.ipynb     # HD-CNN vs baseline, routing analysis
│
├── checkpoints/           # Auto-created — .pt and .pkl files land here
├── figures/               # Auto-created — saved plots land here
└── data/                  # Auto-created — CIFAR-100 downloaded here
```

## Workflow

### 1. Train the baseline (run once)
```bash
python scripts/run_baseline.py
# or override epochs:
python scripts/run_baseline.py --epochs 100
```
Saves to `checkpoints/baseline_best.pt`. **Resumes automatically** if interrupted.

### 2. Run ablations (run once)
```bash
python scripts/run_ablation.py               # all three studies
python scripts/run_ablation.py --study arch  # architecture only
python scripts/run_ablation.py --study reg   # regularization only
python scripts/run_ablation.py --study tta   # TTA only (needs baseline ckpt)
```
Saves to `checkpoints/ablation_results.pkl`. Each experiment also saves its own
`.pt` checkpoint — interrupting and re-running resumes from where it left off.

### 3. Train HD-CNN (run once, after baseline)
```bash
python scripts/run_hdcnn.py                       # uses CIFAR-100 superclasses
python scripts/run_hdcnn.py --hierarchy learned   # derives groups from confusion matrix
```
Saves to `checkpoints/hdcnn_best.pt`. Initializes from baseline weights automatically.

### 4. Explore results in notebooks
Open any notebook in `notebooks/` — cells load the checkpoint and visualize.
No re-training ever happens in notebooks.

## Key fixes vs original notebook

| Issue | Fix |
|---|---|
| `EPOCHS = 50` but paper claims 100 | Unified to 100 in `config.py` |
| `weight_decay` hardcoded in `run_experiment` | Now correctly passed through as parameter |
| No checkpointing — restart = re-train from scratch | Every epoch saves a checkpoint; scripts resume automatically |
| Training + visualization mixed in one notebook | Scripts train; notebooks only load + plot |
| HD-CNN hierarchy derived from confusion matrix but paper uses predefined superclasses | `run_hdcnn.py --hierarchy cifar` uses canonical CIFAR-100 groups (default); `--hierarchy learned` uses confusion-matrix spectral clustering |

## Requirements
```
torch torchvision
scikit-learn
matplotlib
tensorboard
tqdm
```
