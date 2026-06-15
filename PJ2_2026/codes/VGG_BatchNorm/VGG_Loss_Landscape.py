import argparse
import csv
import os
import random
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


DEFAULT_LRS = (1e-3, 2e-3, 1e-4, 5e-4)
PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPORTS_DIR / "models"
TABLES_DIR = REPORTS_DIR / "tables"


def build_device(device_name=None):
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_random_seeds(seed_value=2020, device="cpu"):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            predicted_class = torch.argmax(prediction, dim=1)
            correct += (predicted_class == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train(model, optimizer, criterion, train_loader, val_loader, device,
          scheduler=None, epochs_n=20, best_model_path=None):
    model.to(device)
    batches_n = len(train_loader)
    epoch_losses = []
    step_losses = []
    grad_norms = []
    train_accuracy_curve = []
    val_accuracy_curve = []
    max_val_accuracy = 0.0
    max_val_accuracy_epoch = 0

    for epoch in tqdm(range(epochs_n), unit="epoch"):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            final_weight_grad = model.classifier[4].weight.grad.detach().clone()
            grad_norms.append(final_weight_grad.norm().item())
            step_losses.append(loss.item())
            running_loss += loss.item()

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / batches_n
        epoch_losses.append(epoch_loss)
        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, val_loader, device)
        train_accuracy_curve.append(train_accuracy)
        val_accuracy_curve.append(val_accuracy)

        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)

        print(
            f"epoch={epoch + 1:03d} "
            f"loss={epoch_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

    return {
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "train_accuracy": train_accuracy_curve,
        "val_accuracy": val_accuracy_curve,
        "best_val_accuracy": max_val_accuracy,
        "best_val_accuracy_epoch": max_val_accuracy_epoch + 1,
    }


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_epochs = len(history["epoch_losses"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_accuracy", "val_accuracy"])
        for idx in range(max_epochs):
            writer.writerow([
                idx + 1,
                history["epoch_losses"][idx],
                history["train_accuracy"][idx],
                history["val_accuracy"][idx],
            ])


def plot_training_curves(histories, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history in histories.items():
        epochs = np.arange(1, len(history["epoch_losses"]) + 1)
        axes[0].plot(epochs, history["epoch_losses"], label=label)
        axes[1].plot(epochs, history["val_accuracy"], label=label)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_loss_band(loss_lists):
    min_len = min(len(losses) for losses in loss_lists)
    aligned = np.array([losses[:min_len] for losses in loss_lists])
    return aligned.min(axis=0), aligned.max(axis=0)


def plot_loss_landscape(losses_by_model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, loss_lists in losses_by_model.items():
        min_curve, max_curve = build_loss_band(loss_lists)
        steps = np.arange(len(min_curve))
        ax.plot(steps, min_curve, label=f"{label} min")
        ax.plot(steps, max_curve, label=f"{label} max")
        ax.fill_between(steps, min_curve, max_curve, alpha=0.18)
    ax.set_title("Loss landscape across learning rates")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "learning_rate",
                "parameters",
                "best_val_accuracy",
                "best_val_accuracy_epoch",
                "final_val_accuracy",
                "model_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(args):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    device = build_device(args.device)
    set_random_seeds(args.seed, device=device)
    print(f"Using device: {device}")

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
        download=args.download_data,
    )
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_items,
        download=args.download_data,
    )

    model_classes = {
        "vgg_a": VGG_A,
        "vgg_a_bn": VGG_A_BatchNorm,
    }
    losses_by_model = {name: [] for name in model_classes}
    best_histories = {}
    summary_rows = []

    for model_name, model_cls in model_classes.items():
        best_history = None
        for lr in args.learning_rates:
            set_random_seeds(args.seed, device=device)
            model = model_cls()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            weight_path = MODELS_DIR / f"{model_name}_lr_{lr:g}.pt"
            history = train(
                model,
                optimizer,
                criterion,
                train_loader,
                val_loader,
                device,
                epochs_n=args.epochs,
                best_model_path=weight_path,
            )

            history_path = TABLES_DIR / f"{model_name}_lr_{lr:g}_history.csv"
            save_history(history, history_path)
            np.savetxt(TABLES_DIR / f"{model_name}_lr_{lr:g}_step_losses.txt", history["step_losses"])
            np.savetxt(TABLES_DIR / f"{model_name}_lr_{lr:g}_grad_norms.txt", history["grad_norms"])
            losses_by_model[model_name].append(history["step_losses"])

            summary_rows.append({
                "model": model_name,
                "learning_rate": lr,
                "parameters": get_number_of_parameters(model),
                "best_val_accuracy": history["best_val_accuracy"],
                "best_val_accuracy_epoch": history["best_val_accuracy_epoch"],
                "final_val_accuracy": history["val_accuracy"][-1],
                "model_path": str(weight_path),
            })

            if best_history is None or history["best_val_accuracy"] > best_history["best_val_accuracy"]:
                best_history = history

        best_histories[model_name] = best_history

    write_summary(summary_rows, TABLES_DIR / "summary.csv")
    plot_training_curves(best_histories, FIGURES_DIR / "training_curves.png")
    plot_loss_landscape(losses_by_model, FIGURES_DIR / "loss_landscape_comparison.png")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved tables to {TABLES_DIR}")
    print(f"Saved model weights to {MODELS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG-A and VGG-A-BN on CIFAR-10.")
    parser.add_argument("--data-root", default="./data", help="CIFAR-10 data directory.")
    parser.add_argument("--download-data", action="store_true",
                        help="Download CIFAR-10 if it is not already under --data-root.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-items", type=int, default=-1, help="Limit train samples; -1 uses all.")
    parser.add_argument("--val-items", type=int, default=-1, help="Limit validation samples; -1 uses all.")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda:0.")
    parser.add_argument("--quick", action="store_true", help="Small smoke test for debugging.")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 1
        args.batch_size = 64
        args.num_workers = 0
        args.n_items = 512
        args.val_items = 256
        args.learning_rates = [1e-3, 2e-3]
    return args


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    run_experiments(parse_args())
