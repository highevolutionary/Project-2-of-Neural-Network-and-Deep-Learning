# Project 2 Running Guide

This project contains the completed CIFAR-10 experiments and the VGG-A Batch Normalization comparison.

## Data

Put the CIFAR-10 Python archive under:

```bash
PJ2_2026/codes/VGG_BatchNorm/data/cifar-10-python.tar.gz
```

Then extract it in the same directory:

```bash
cd PJ2_2026/codes/VGG_BatchNorm/data
tar -xzf cifar-10-python.tar.gz
cd ../../..
```

The expected folder is:

```bash
PJ2_2026/codes/VGG_BatchNorm/data/cifar-10-batches-py
```

## Full Experiment

From the `PJ2_2026` folder, run:

```bash
nohup python run_project2.py --device cuda:0 > train.log 2>&1 &
```

## Quick Check

```bash
python run_project2.py --quick --device cuda:0
```

## Outputs

The script saves figures, tables, and model checkpoints under:

```bash
pj2_outputs/
```
