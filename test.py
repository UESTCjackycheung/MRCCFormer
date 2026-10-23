"""Evaluate a saved checkpoint on its held-out test or validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from dataset import DATASETS, get_data_loaders
from model import build_model, count_parameters
from train import resolve_device, write_csv, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--weight-path", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override the data location stored in the checkpoint",
    )
    parser.add_argument("--cache-dir", type=Path, help="HisarMod NPY cache directory")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Optional consistency check against the saved dataset",
    )
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


@torch.inference_mode()
def infer(model, loader, device, progress=True):
    model.eval()
    labels_all, predictions_all, snrs_all = [], [], []
    loss_sum = 0.0
    for signals, labels, snrs in tqdm(
        loader, desc="Evaluation", disable=not progress, leave=False
    ):
        signals = signals.to(device, dtype=torch.float32, non_blocking=True)
        targets = labels.to(device, dtype=torch.long, non_blocking=True)
        logits = model(signals)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model output")
        loss_sum += torch.nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        labels_all.append(labels.numpy())
        predictions_all.append(logits.argmax(1).cpu().numpy())
        snrs_all.append(snrs.numpy())
    if not labels_all:
        raise ValueError("Evaluation split is empty")
    labels = np.concatenate(labels_all)
    return (
        labels,
        np.concatenate(predictions_all),
        np.concatenate(snrs_all),
        loss_sum / labels.size,
    )


def calculate_results(labels, predictions, snrs, class_names):
    class_ids = np.arange(len(class_names))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    expected = (
        np.sum(matrix.sum(axis=0).astype(float) * matrix.sum(axis=1)) / labels.size**2
    )
    kappa = (
        float(cohen_kappa_score(labels, predictions, labels=class_ids))
        if expected < 1.0
        else None
    )
    overall = {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=class_ids, average="macro", zero_division=0
            )
        ),
        "cohen_kappa": kappa,
    }
    by_snr, by_class, by_class_snr = [], [], []
    for snr in np.unique(snrs):
        mask = snrs == snr
        by_snr.append(
            dict(
                snr=int(snr),
                samples=int(mask.sum()),
                accuracy=float(np.mean(labels[mask] == predictions[mask])),
            )
        )
    for class_id, name in enumerate(class_names):
        mask = labels == class_id
        by_class.append(
            dict(
                class_id=class_id,
                modulation=name,
                samples=int(mask.sum()),
                accuracy=float(np.mean(predictions[mask] == class_id))
                if mask.any()
                else None,
            )
        )
        for snr in np.unique(snrs):
            cell = mask & (snrs == snr)
            by_class_snr.append(
                dict(
                    class_id=class_id,
                    modulation=name,
                    snr=int(snr),
                    samples=int(cell.sum()),
                    accuracy=float(np.mean(predictions[cell] == class_id))
                    if cell.any()
                    else None,
                )
            )
    return overall, by_snr, by_class, by_class_snr, matrix


def save_plots(matrix, classes, by_snr, output_dir):
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    side = max(7, len(classes) * 0.42)
    fig, ax = plt.subplots(figsize=(side, side))
    display = ax.imshow(
        normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
    )
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted modulation",
        ylabel="True modulation",
    )
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(
        display, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [row["snr"] for row in by_snr], [row["accuracy"] for row in by_snr], marker="o"
    )
    ax.set(xlabel="SNR (dB)", ylabel="Accuracy", ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_snr.png", dpi=200)
    plt.close(fig)
    return normalized


def evaluate(args):
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Use a checkpoint produced by this package's train.py")
    config, saved_metadata = checkpoint["config"], checkpoint["metadata"]
    if args.dataset is not None and args.dataset != config["dataset"]:
        raise ValueError("--dataset does not match the checkpoint")
    path = args.data_path or Path(config["data_path"])
    cache = args.cache_dir or (
        Path(config["cache_dir"]) if config.get("cache_dir") else None
    )
    _, val_loader, test_loader, metadata = get_data_loaders(
        path,
        dataset=config["dataset"],
        batch_size=config["batch_size"],
        eval_batch_size=args.batch_size,
        seed=config["seed"],
        num_workers=args.num_workers,
        cache_dir=cache,
    )
    if metadata != saved_metadata:
        raise ValueError(
            "Dataset classes, preprocessing, or split differs from the saved checkpoint"
        )
    model = build_model(
        config["model"], metadata["num_classes"], metadata["signal_length"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = test_loader if args.split == "test" else val_loader
    labels, predictions, snrs, loss = infer(model, loader, device, not args.quiet)
    overall, by_snr, by_class, by_class_snr, matrix = calculate_results(
        labels, predictions, snrs, metadata["mods"]
    )
    overall.update(
        loss=loss,
        dataset=config["dataset"],
        model=config["model"],
        seed=config["seed"],
        checkpoint_epoch=checkpoint["epoch"],
        split=args.split,
        parameters=count_parameters(model)[0],
        split_digest=metadata["split_digest"],
    )
    output_dir = args.output_dir or args.checkpoint.resolve().parent / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(overall, output_dir / "metrics.json")
    write_json(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "data_path": str(path.resolve()),
            "device": str(device),
            "batch_size": args.batch_size,
        },
        output_dir / "evaluation.json",
    )
    write_csv(by_snr, output_dir / "accuracy_by_snr.csv")
    write_csv(by_class, output_dir / "accuracy_by_class.csv")
    write_csv(by_class_snr, output_dir / "accuracy_by_class_snr.csv")
    write_csv(
        [
            dict(
                modulation=name,
                **{
                    other: int(value)
                    for other, value in zip(metadata["mods"], row, strict=True)
                },
            )
            for name, row in zip(metadata["mods"], matrix, strict=True)
        ],
        output_dir / "confusion_matrix.csv",
    )
    normalized = save_plots(matrix, metadata["mods"], by_snr, output_dir)
    np.save(
        output_dir / "confusion_matrix_normalized.npy", normalized, allow_pickle=False
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=labels,
        predictions=predictions,
        snrs=snrs,
        source_indices=loader.dataset.indices,
    )
    print(
        f"{args.split.capitalize()} samples: {overall['samples']}; accuracy: {overall['accuracy']:.4f}; macro F1: {overall['macro_f1']:.4f}",
        flush=True,
    )
    print(f"Results: {output_dir.resolve()}", flush=True)
    return overall


if __name__ == "__main__":
    evaluate(parse_args())
