"""Train and validate the proposed model or a comparison method."""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset import DATASETS, default_data_path, get_data_loaders
from model import MODEL_NAMES, build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name="auto") -> torch.device:
    device = (
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, and cuda:<index>")
    if device.type == "cuda":
        if (
            not torch.cuda.is_available()
            or (device.index or 0) >= torch.cuda.device_count()
        ):
            raise ValueError(f"CUDA device is unavailable: {device}")
        torch.cuda.set_device(device)
    return device


def random_iq_rotation(signals, max_degrees=45.0):
    angles = signals.new_empty(signals.shape[0], 1).uniform_(
        -math.radians(max_degrees), math.radians(max_degrees)
    )
    cosine, sine = (angles.cos(), angles.sin())
    i, q = signals.unbind(dim=1)
    return torch.stack((cosine * i - sine * q, sine * i + cosine * q), dim=1)


def run_epoch(
    model, loader, criterion, device, optimizer=None, rotation=False, progress=True
):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for signals, labels, _ in tqdm(
            loader,
            desc="Train" if training else "Validation",
            disable=not progress,
            leave=False,
        ):
            signals = signals.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            if training and rotation:
                signals = random_iq_rotation(signals)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; training was stopped")
            if training:
                loss.backward()
                finite = [
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ]
                if finite and (not torch.stack(finite).all()):
                    raise FloatingPointError(
                        "Non-finite gradient; optimizer step was skipped"
                    )
                optimizer.step()
            size = labels.size(0)
            loss_sum += loss.item() * size
            correct += (logits.argmax(1) == labels).sum().item()
            count += size
    if not count:
        raise ValueError("The data loader is empty")
    return (loss_sum / count, correct / count)


@dataclass
class EarlyStopping:
    """Stop after ten epochs without a new minimum validation loss."""

    patience: int = 10
    best: float = math.inf
    bad_epochs: int = 0

    def step(self, loss: float) -> bool:
        if not math.isfinite(loss):
            raise ValueError("Validation loss must be finite")
        if loss < self.best:
            self.best, self.bad_epochs = (loss, 0)
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience


def save_checkpoint(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def write_json(data, path):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(rows, path):
    if not rows:
        raise ValueError("Cannot write an empty table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def make_scheduler(optimizer):
    """Halve the rate after three epochs without a new minimum validation loss."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=0.0,
        threshold_mode="abs",
        cooldown=0,
        min_lr=0.0,
        eps=0.0,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="rml2016a")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model", choices=MODEL_NAMES, default="MRPureCumFormer")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 0 <= args.seed < 2**32
        or min(args.epochs, args.batch_size, args.eval_batch_size) < 1
        or args.num_workers < 0
    ):
        parser.error("Invalid seed, epochs, batch sizes, or worker count")
    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("Require finite lr > 0 and weight-decay >= 0")
    return args


def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    path = (args.data_path or default_data_path(args.dataset)).resolve()
    results = Path(__file__).resolve().parent / "results"
    output = args.output_dir or results / args.dataset / args.model / str(args.seed)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    print(f"Loading {args.dataset}: {path}", flush=True)
    train_loader, val_loader, _, metadata = get_data_loaders(
        path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    model = build_model(
        args.model, metadata["num_classes"], metadata["signal_length"]
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer)
    stopping = EarlyStopping()
    criterion = nn.CrossEntropyLoss()
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(data_path=str(path), output_dir=str(output.resolve()))
    output.mkdir(parents=True, exist_ok=True)
    write_json(config, output / "config.json")
    history = []
    best_accuracy, best_accuracy_loss = -1.0, math.inf
    print(
        f"{args.model}: {count_parameters(model)[0]:,} parameters; device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            rotation=True,
            progress=not args.quiet,
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, progress=not args.quiet
        )
        scheduler.step(val_loss)
        stop = stopping.step(val_loss)
        history.append(
            dict(
                epoch=epoch,
                learning_rate=rate,
                next_learning_rate=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
        )
        improved = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_accuracy_loss
        )
        if improved:
            best_accuracy, best_accuracy_loss = val_accuracy, val_loss
            save_checkpoint(
                {
                    "format_version": 2,
                    "epoch": epoch,
                    "config": config,
                    "metadata": metadata,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "early_stopping": asdict(stopping),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                },
                output / "best.pt",
            )
        write_csv(history, output / "history.csv")
        print(
            f"Epoch {epoch:03d}: train loss={train_loss:.5f}, accuracy={train_accuracy:.4f}; val loss={val_loss:.5f}, accuracy={val_accuracy:.4f}; lr={rate:.3g}",
            flush=True,
        )
        if optimizer.param_groups[0]["lr"] < rate:
            print(
                f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.3g}",
                flush=True,
            )
        if stop:
            print(
                "Early stopping: validation loss did not decrease for ten consecutive epochs.",
                flush=True,
            )
            break
    print(
        f"Best validation accuracy: {best_accuracy:.4f}; checkpoint: {output.resolve() / 'best.pt'}",
        flush=True,
    )
    return {"best_val_accuracy": best_accuracy, "epochs": len(history)}


if __name__ == "__main__":
    train(parse_args())
