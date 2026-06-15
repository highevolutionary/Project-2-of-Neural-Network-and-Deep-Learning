# GPU Server Run Guide

The code is portable to a Linux GPU server. There are no Windows-only paths in
the training script, and the default device automatically becomes `cuda:0` when
CUDA is available.

## 1. Create an environment

```bash
conda create -n pj2 python=3.11 -y
conda activate pj2
```

Install the PyTorch build matching the server CUDA version. The official PyTorch
selector is here:

https://pytorch.org/get-started/locally/

Example for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib numpy tqdm
```

Example for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib numpy tqdm
```

## 2. Upload the project

Upload the whole `PJ2_2026` directory to the server. Then run:

```bash
cd PJ2_2026/codes/VGG_BatchNorm
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

The bundled `data/cifar-10-python.tar.gz` in the assignment archive appears to
be truncated. If the server has network access, `torchvision` will download a
fresh CIFAR-10 copy automatically. If the server has no network access, upload a
complete CIFAR-10 Python archive or an extracted `cifar-10-batches-py` directory
into `PJ2_2026/codes/VGG_BatchNorm/data`.

## 3. Smoke test

```bash
python VGG_Loss_Landscape.py --quick --device cuda:0
```

This should produce files under `reports/`.

## 4. Full run

```bash
nohup python VGG_Loss_Landscape.py \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda:0 \
  > train.log 2>&1 &
```

If the server GPU memory is small, reduce the batch size:

```bash
python VGG_Loss_Landscape.py --epochs 20 --batch-size 64 --device cuda:0
```

## 5. Outputs for the report

Use these files in the final PDF:

- `reports/figures/training_curves.png`
- `reports/figures/loss_landscape_comparison.png`
- `reports/tables/summary.csv`
- `reports/models/*.pt`

Upload the model weights and dataset to netdisk or Google Drive, then paste
those links in the report.
