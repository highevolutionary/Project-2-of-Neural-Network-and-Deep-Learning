"""Project 2 training script with controlled optimization comparisons."""

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
BASE_DROPOUT = 0.4
BASE_MAX_LR = 8e-4
BASE_WEIGHT_DECAY = 3e-4
BASE_GRAD_CLIP = 0.8
BASE_NUM_BLOCKS = 6
OUTPUT_DIR = Path("pj2_reworked_outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def set_random_seeds(seed=22300290002, device="cpu"):
    np.random.seed(seed % (2 ** 32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=3, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.10),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def get_cifar_loader(root, batch_size, train=True, shuffle=True,
                     num_workers=4, n_items=-1, download=False):
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=get_transforms(train=train),
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = make_activation(activation)
        self.se = SEModule(out_channels)

    def forward(self, x):
        out = self.main(x) + self.shortcut(x)
        out = self.activation(out)
        return self.se(out)


class CourseCNN(nn.Module):
    def __init__(self, activation="relu", num_blocks=BASE_NUM_BLOCKS,
                 dropout=BASE_DROPOUT, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(24),
            make_activation(activation),
        )
        self.stage1 = self._make_stage(24, 24, num_blocks, activation, stride=1)
        self.stage2 = self._make_stage(24, 32, num_blocks, activation, stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks, activation, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights(activation)

    def _make_stage(self, in_channels, out_channels, num_blocks, activation, stride):
        blocks = [BasicBlock(in_channels, out_channels, activation=activation, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_channels, out_channels, activation=activation, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self, activation):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if activation == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                if activation == "relu":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class VGG_A(nn.Module):
    def __init__(self, use_bn=False, num_classes=10):
        super().__init__()

        def conv(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.features = nn.Sequential(
            *conv(3, 64),
            nn.MaxPool2d(2, 2),
            *conv(64, 128),
            nn.MaxPool2d(2, 2),
            *conv(128, 256),
            *conv(256, 256),
            nn.MaxPool2d(2, 2),
            *conv(256, 512),
            *conv(512, 512),
            nn.MaxPool2d(2, 2),
            *conv(512, 512),
            *conv(512, 512),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class WarmupCosineScheduler:
    def __init__(self, optimizer, max_lr, epochs, warmup_epochs=5):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
            lr = 0.5 * self.max_lr * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def build_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, device, epochs, optimizer_name,
                lr, weight_decay, grad_clip, patience, min_delta, out_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, max_lr=lr, epochs=epochs)
    history = []
    best_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in tqdm(range(epochs), unit="epoch"):
        current_lr = scheduler.step(epoch)
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if epoch >= scheduler.warmup_epochs:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc = accuracy(model, test_loader, device)
        epoch_loss = total_loss / len(train_loader)
        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "loss": epoch_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })
        print(
            f"epoch={epoch + 1:03d} lr={current_lr:.6g} "
            f"loss={epoch_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )
        if test_acc > best_acc + min_delta:
            best_acc = test_acc
            best_epoch = epoch + 1
            stale_epochs = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            break
    return history, best_acc, best_epoch


def train_vgg_landscape(model, train_loader, test_loader, device, epochs, lr, use_bn, out_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_losses = []
    history = []
    best_acc = 0.0
    best_epoch = 0
    for epoch in tqdm(range(epochs), unit="epoch"):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        test_acc = accuracy(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "train_accuracy": correct / total,
            "test_accuracy": test_acc,
            "learning_rate": lr,
            "use_bn": use_bn,
        })
        print(
            f"epoch={epoch + 1:03d} loss={history[-1]['loss']:.4f} "
            f"train_acc={history[-1]['train_accuracy']:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )
    return history, step_losses, best_acc, best_epoch


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_accuracy(histories, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, history in histories.items():
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_accuracy"] for row in history], linestyle="--", label=f"{label} train")
        ax.plot(epochs, [row["test_accuracy"] for row in history], label=f"{label} test")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loss_landscape(losses_by_model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, loss_lists in losses_by_model.items():
        min_len = min(len(values) for values in loss_lists)
        aligned = np.array([values[:min_len] for values in loss_lists])
        min_curve = aligned.min(axis=0)
        max_curve = aligned.max(axis=0)
        steps = np.arange(min_len)
        ax.plot(steps, min_curve, label=f"{label} min")
        ax.plot(steps, max_curve, label=f"{label} max")
        ax.fill_between(steps, min_curve, max_curve, alpha=0.18)
    ax.set_title("Loss Landscape Comparison")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss Value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def visualize_filters(model, path):
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module.weight.detach().cpu()
            break
    if first_conv is None:
        return
    filters = first_conv[:24].mean(dim=1)
    fig, axes = plt.subplots(4, 6, figsize=(7, 5))
    for ax, filt in zip(axes.flat, filters):
        arr = filt.numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        ax.imshow(arr, cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_course_experiments(args, device, train_loader, test_loader):
    baseline = {
        "activation": "relu",
        "num_blocks": BASE_NUM_BLOCKS,
        "weight_decay": BASE_WEIGHT_DECAY,
        "optimizer": "adam",
    }
    groups = {
        "main": [("baseline", baseline)],
        "activation": [
            ("relu", {**baseline, "activation": "relu"}),
            ("tanh", {**baseline, "activation": "tanh"}),
            ("sigmoid", {**baseline, "activation": "sigmoid"}),
        ],
        "structure": [
            ("blocks_5", {**baseline, "num_blocks": 5}),
            ("blocks_6", {**baseline, "num_blocks": 6}),
            ("blocks_7", {**baseline, "num_blocks": 7}),
        ],
        "loss": [
            ("wd_5e-5", {**baseline, "weight_decay": 5e-5}),
            ("wd_3e-4", {**baseline, "weight_decay": 3e-4}),
            ("wd_8e-4", {**baseline, "weight_decay": 8e-4}),
        ],
        "optimizer": [
            ("adam", {**baseline, "optimizer": "adam"}),
            ("sgd", {**baseline, "optimizer": "sgd"}),
        ],
    }
    rows = []
    for group_name, experiments in groups.items():
        histories = {}
        for label, config in experiments:
            exp_name = f"{group_name}_{label}"
            exp_epochs = args.baseline_epochs if exp_name == "main_baseline" else args.epochs
            print(f"\nTraining {exp_name}")
            set_random_seeds(args.seed, device=device)
            model = CourseCNN(
                activation=config["activation"],
                num_blocks=config["num_blocks"],
                dropout=BASE_DROPOUT,
            )
            history, best_acc, best_epoch = train_model(
                model,
                train_loader,
                test_loader,
                device=device,
                epochs=exp_epochs,
                optimizer_name=config["optimizer"],
                lr=BASE_MAX_LR,
                weight_decay=config["weight_decay"],
                grad_clip=BASE_GRAD_CLIP,
                patience=args.patience,
                min_delta=args.min_delta,
                out_path=MODEL_DIR / f"{exp_name}.pt",
            )
            save_history(history, TABLE_DIR / f"{exp_name}_history.csv")
            histories[label] = history
            rows.append({
                "experiment": exp_name,
                "group": group_name,
                "activation": config["activation"],
                "num_blocks": config["num_blocks"],
                "optimizer": config["optimizer"],
                "learning_rate": BASE_MAX_LR,
                "weight_decay": config["weight_decay"],
                "dropout": BASE_DROPOUT,
                "gradient_clip": BASE_GRAD_CLIP,
                "parameters": count_params(model),
                "best_test_accuracy": best_acc,
                "best_epoch": best_epoch,
                "final_test_accuracy": history[-1]["test_accuracy"],
            })
            if exp_name == "main_baseline":
                visualize_filters(model, FIGURE_DIR / "first_layer_filters.png")
        if group_name != "main":
            plot_accuracy(histories, f"Accuracy with Different {group_name.title()}", FIGURE_DIR / f"{group_name}_comparison.png")
    with (TABLE_DIR / "course_cnn_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_vgg_bn_experiments(args, device, train_loader, test_loader):
    lrs = [1e-3, 2e-3, 1e-4, 5e-4]
    rows = []
    losses_by_model = {"VGG-A": [], "VGG-A-BN": []}
    best_histories = {}
    for use_bn in [False, True]:
        model_label = "VGG-A-BN" if use_bn else "VGG-A"
        best_acc_for_model = -1
        for lr in lrs:
            exp_name = f"{'vgg_a_bn' if use_bn else 'vgg_a'}_lr_{lr:g}"
            print(f"\nTraining {exp_name}")
            set_random_seeds(args.seed, device=device)
            model = VGG_A(use_bn=use_bn)
            history, step_losses, best_acc, best_epoch = train_vgg_landscape(
                model,
                train_loader,
                test_loader,
                device=device,
                epochs=args.vgg_epochs,
                lr=lr,
                use_bn=use_bn,
                out_path=MODEL_DIR / f"{exp_name}.pt",
            )
            save_history(history, TABLE_DIR / f"{exp_name}_history.csv")
            np.savetxt(TABLE_DIR / f"{exp_name}_step_losses.txt", step_losses)
            losses_by_model[model_label].append(step_losses)
            rows.append({
                "model": model_label,
                "learning_rate": lr,
                "parameters": count_params(model),
                "best_test_accuracy": best_acc,
                "best_epoch": best_epoch,
                "final_test_accuracy": history[-1]["test_accuracy"],
            })
            if best_acc > best_acc_for_model:
                best_acc_for_model = best_acc
                best_histories[model_label] = history
    with (TABLE_DIR / "vgg_bn_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_accuracy(best_histories, "VGG-A with and without Batch Normalization", FIGURE_DIR / "vgg_bn_training_comparison.png")
    plot_loss_landscape(losses_by_model, FIGURE_DIR / "vgg_loss_landscape_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Project 2 experiments.")
    parser.add_argument("--data-root", default="codes/VGG_BatchNorm/data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--baseline-epochs", type=int, default=50)
    parser.add_argument("--vgg-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=192)
    parser.add_argument("--test-batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=0.0007)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--val-items", type=int, default=-1)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--suite", choices=["all", "course", "vgg"], default="all")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.baseline_epochs = 2
        args.vgg_epochs = 1
        args.n_items = 1024
        args.val_items = 480
        args.num_workers = 0
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    set_random_seeds(args.seed, device=device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_loader = get_cifar_loader(
        args.data_root,
        batch_size=args.train_batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
        download=args.download_data,
    )
    test_loader = get_cifar_loader(
        args.data_root,
        batch_size=args.test_batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.val_items,
        download=args.download_data,
    )
    if args.suite in ("all", "course"):
        run_course_experiments(args, device, train_loader, test_loader)
    if args.suite in ("all", "vgg"):
        run_vgg_bn_experiments(args, device, train_loader, test_loader)
    print(f"Saved outputs to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
