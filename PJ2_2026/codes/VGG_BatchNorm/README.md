# Project 2: CIFAR-10 VGG-A and Batch Normalization

This folder contains a reproducible PyTorch implementation for the required
CIFAR-10 experiments:

- `VGG_A`: baseline VGG-A network for 32x32 CIFAR-10 images.
- `VGG_A_BatchNorm`: VGG-A with `BatchNorm2d` after each convolution.
- Training with multiple learning rates.
- Validation accuracy, model weights, per-step losses, gradient norms, and loss
  landscape comparison figures.

## Environment

Install PyTorch and torchvision in a Python environment. Example:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

The supplied archive already includes `data/cifar-10-python.tar.gz`. The
torchvision loader can also download CIFAR-10 automatically if the archive is
missing and network access is available.

## Quick Smoke Test

Run this first to verify the code path:

```bash
python VGG_Loss_Landscape.py --quick --device cpu
```

The smoke test uses a tiny subset and one epoch, so its accuracy is not
meaningful. It only checks that data loading, training, saving, and plotting
work.

## Full Experiment

Recommended GPU run:

```bash
python VGG_Loss_Landscape.py --epochs 20 --batch-size 128 --device cuda:0
```

To use only part of CIFAR-10 for a faster draft:

```bash
python VGG_Loss_Landscape.py --epochs 10 --n-items 10000 --val-items 2000 --device cuda:0
```

Default learning rates are `1e-3 2e-3 1e-4 5e-4`, matching the assignment
suggestion. They can be changed:

```bash
python VGG_Loss_Landscape.py --learning-rates 0.001 0.0005 0.0001
```

## Outputs

Running the script creates:

- `reports/figures/training_curves.png`
- `reports/figures/loss_landscape_comparison.png`
- `reports/tables/summary.csv`
- `reports/tables/*_history.csv`
- `reports/tables/*_step_losses.txt`
- `reports/tables/*_grad_norms.txt`
- `reports/models/*.pt`

Include the figures in the report. Upload the trained `*.pt` weights and the
CIFAR-10 dataset archive to a netdisk or Google Drive, then paste both links in
the final PDF as required by the teacher.

## Report Checklist

- Name and student ID.
- GitHub link to this code.
- Dataset link.
- Trained model weights link.
- Best test or validation error.
- Network structures and parameter counts.
- Required components: fully connected layer, 2D convolution, 2D pooling,
  activation, and BatchNorm.
- Optimization attempts: different filter counts or architectures, different
  regularization/loss settings, different activations, and optimizer choice.
- BN comparison: VGG-A versus VGG-A-BN training curve and accuracy.
- Insight visualization: loss landscape comparison and discussion.
