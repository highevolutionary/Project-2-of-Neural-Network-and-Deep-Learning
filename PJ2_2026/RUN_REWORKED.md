# Reworked Project 2 Run Commands

This version keeps the same experiment categories as the reference report, but
uses different numerical settings and the full CIFAR-10 train/test split by
default.

## Data

The CIFAR-10 archive is placed at:

```text
codes/VGG_BatchNorm/data/cifar-10-python.tar.gz
```

On the server:

```bash
cd PJ2_2026
cd codes/VGG_BatchNorm/data
tar -xzf cifar-10-python.tar.gz
cd ../../..
```

## Quick Test

```bash
python run_reworked_project2.py --quick --device cuda:0
```

## Full Run

```bash
nohup python run_reworked_project2.py --device cuda:0 > train_reworked.log 2>&1 &
```

Outputs are saved to:

```text
pj2_reworked_outputs/
```

## Main Differences From The Reference Settings

- Train batch size: 192
- Test batch size: 384
- Random crop padding: 3
- Color jitter: brightness/contrast/saturation 0.25, hue 0.10
- Dropout rate: 0.4
- Max learning rate: 0.0008
- Weight decay baseline: 0.0003
- Gradient clip: 0.8
- Early stopping patience: 16
- Early stopping min delta: 0.0007
- Structure comparison: 5, 6, and 7 blocks
- Loss/regularization comparison: 5e-5, 3e-4, and 8e-4 weight decay
