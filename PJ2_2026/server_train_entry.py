"""
Temporary server entry for Project 2 training.

Put this file at the root of the uploaded PJ2_2026 folder, run it from the
terminal, then delete it after training if you only want to keep the completed
assignment code in codes/VGG_BatchNorm on GitHub.

Expected offline CIFAR-10 layout:
    PJ2_2026/codes/VGG_BatchNorm/data/cifar-10-batches-py/

Example:
    python server_train_entry.py --quick --device cuda:0
    nohup python server_train_entry.py --epochs 20 --batch-size 128 --device cuda:0 > train.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT / "codes" / "VGG_BatchNorm"
TRAIN_SCRIPT = WORKDIR / "VGG_Loss_Landscape.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary launcher for the completed Project 2 code.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--learning-rates", nargs="+", default=["0.001", "0.002", "0.0001", "0.0005"])
    return parser.parse_args()


def main():
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find training script: {TRAIN_SCRIPT}")

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--data-root", args.data_root,
        "--learning-rates", *args.learning_rates,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.quick:
        cmd.append("--quick")
    if args.download_data:
        cmd.append("--download-data")

    print("Running:")
    print(" ".join(cmd))
    print(f"Working directory: {WORKDIR}")
    os.chdir(WORKDIR)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
