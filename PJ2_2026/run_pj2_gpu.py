"""Training script for the CIFAR-10 experiments in Project 2."""

import argparse
import csv
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


DEFAULT_LRS = (1e-3, 2e-3, 1e-4, 5e-4)
OUTPUT_DIR = Path("pj2_outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root="./data", batch_size=128, train=True, shuffle=True,
                     num_workers=4, n_items=-1, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    try:
        dataset = datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
    except RuntimeError as exc:
        if download:
            dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        else:
            raise RuntimeError(
                "CIFAR-10 was not found locally and automatic download is disabled.\n"
                "Expected structure:\n"
                f"  {Path(root).resolve()}/cifar-10-batches-py/data_batch_1\n"
                f"  {Path(root).resolve()}/cifar-10-batches-py/data_batch_2\n"
                "  ...\n\n"
                "Fix one of these ways:\n"
                "  1. Upload and extract the complete CIFAR-10 Python dataset into ./data, or\n"
                "  2. Run with --download-data on a server that can access the internet.\n"
            ) from exc
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def init_weights_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_number_of_parameters(model):
    return sum(np.prod(parameter.shape).item() for parameter in model.parameters())


class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self.apply(init_weights_)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self.apply(init_weights_)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def make_activation(name):
    if name == "relu":
        return nn.ReLU(True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def scaled_channels(width_mult):
    base = [64, 128, 256, 512, 512]
    return [max(16, int(channel * width_mult)) for channel in base]


class ConfigurableVGG(nn.Module):
    def __init__(self, width_mult=1.0, activation="relu", use_bn=True,
                 dropout=0.0, inp_ch=3, num_classes=10):
        super().__init__()
        c1, c2, c3, c4, c5 = scaled_channels(width_mult)

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(make_activation(activation))
            return layers

        self.features = nn.Sequential(
            *conv_block(inp_ch, c1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *conv_block(c1, c2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *conv_block(c2, c3),
            *conv_block(c3, c3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *conv_block(c3, c4),
            *conv_block(c4, c4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            *conv_block(c4, c5),
            *conv_block(c5, c5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        classifier_layers = [nn.Linear(c5, 512), make_activation(activation)]
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.extend([
            nn.Linear(512, 512),
            make_activation(activation),
        ])
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(512, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.apply(init_weights_)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prediction = model(x)
            predicted_class = torch.argmax(prediction, dim=1)
            correct += (predicted_class == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train(model, optimizer, criterion, train_loader, val_loader, device,
          epochs_n=20, best_model_path=None):
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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()

            grad_norms.append(model.classifier[4].weight.grad.detach().norm().item())
            step_losses.append(loss.item())
            running_loss += loss.item()
            optimizer.step()

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
            f"val_acc={val_accuracy:.4f}",
            flush=True,
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "train_accuracy", "val_accuracy"])
        for idx, loss in enumerate(history["epoch_losses"]):
            writer.writerow([
                idx + 1,
                loss,
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


def make_optimizer(name, model, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def run_ablation_experiments(args, train_loader, val_loader, device):
    if args.skip_ablation:
        return

    ablations = [
        {
            "name": "filters_width_0.5",
            "category": "different_filters",
            "model_kwargs": {"width_mult": 0.5, "activation": "relu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
        },
        {
            "name": "filters_width_1.0",
            "category": "different_filters",
            "model_kwargs": {"width_mult": 1.0, "activation": "relu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
        },
        {
            "name": "activation_leaky_relu",
            "category": "different_activations",
            "model_kwargs": {"width_mult": 1.0, "activation": "leaky_relu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
        },
        {
            "name": "activation_elu",
            "category": "different_activations",
            "model_kwargs": {"width_mult": 1.0, "activation": "elu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
        },
        {
            "name": "regularization_no_weight_decay",
            "category": "different_loss_regularization",
            "model_kwargs": {"width_mult": 1.0, "activation": "relu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
        },
        {
            "name": "loss_label_smoothing_0.1",
            "category": "different_loss_regularization",
            "model_kwargs": {"width_mult": 1.0, "activation": "relu", "use_bn": True},
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 5e-4,
            "label_smoothing": 0.1,
        },
        {
            "name": "optimizer_sgd_momentum",
            "category": "different_optimizers",
            "model_kwargs": {"width_mult": 1.0, "activation": "relu", "use_bn": True},
            "optimizer": "sgd",
            "lr": 1e-2,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
        },
    ]

    rows = []
    for config in ablations:
        print(f"\nAblation: {config['name']}")
        set_random_seeds(args.seed, device=device)
        model = ConfigurableVGG(**config["model_kwargs"])
        optimizer = make_optimizer(
            config["optimizer"],
            model,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        weight_path = MODELS_DIR / f"ablation_{config['name']}.pt"
        history = train(
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            device,
            epochs_n=args.ablation_epochs,
            best_model_path=weight_path,
        )
        save_history(history, TABLES_DIR / f"ablation_{config['name']}_history.csv")
        rows.append({
            "experiment": config["name"],
            "category": config["category"],
            "parameters": get_number_of_parameters(model),
            "activation": config["model_kwargs"]["activation"],
            "width_mult": config["model_kwargs"]["width_mult"],
            "use_bn": config["model_kwargs"]["use_bn"],
            "optimizer": config["optimizer"],
            "learning_rate": config["lr"],
            "weight_decay": config["weight_decay"],
            "label_smoothing": config["label_smoothing"],
            "best_val_accuracy": history["best_val_accuracy"],
            "best_val_accuracy_epoch": history["best_val_accuracy_epoch"],
            "final_val_accuracy": history["val_accuracy"][-1],
            "model_path": str(weight_path),
        })

    path = TABLES_DIR / "ablation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation summary to: {path.resolve()}")


def run_experiments(args):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    device = build_device(args.device)
    set_random_seeds(args.seed, device=device)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

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
            print(f"\nTraining {model_name} with lr={lr}")
            set_random_seeds(args.seed, device=device)
            model = model_cls()
            optimizer = make_optimizer(args.optimizer, model, lr=lr, weight_decay=args.weight_decay)
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

            save_history(history, TABLES_DIR / f"{model_name}_lr_{lr:g}_history.csv")
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
    run_ablation_experiments(args, train_loader, val_loader, device)
    print(f"\nSaved figures to: {FIGURES_DIR.resolve()}")
    print(f"Saved tables to: {TABLES_DIR.resolve()}")
    print(f"Saved model weights to: {MODELS_DIR.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Project 2 CIFAR-10 runner.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download-data", action="store_true",
                        help="Download CIFAR-10 if it is not already under --data-root.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-items", type=int, default=-1, help="Limit train samples; -1 uses all.")
    parser.add_argument("--val-items", type=int, default=-1, help="Limit validation samples; -1 uses all.")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=list(DEFAULT_LRS))
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda:0, cuda:1.")
    parser.add_argument("--quick", action="store_true", help="Run a short test.")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 1
        args.batch_size = 64
        args.num_workers = 2
        args.n_items = 512
        args.val_items = 256
        args.learning_rates = [1e-3, 2e-3]
        args.ablation_epochs = 1
    return args


if __name__ == "__main__":
    run_experiments(parse_args())
