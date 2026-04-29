"""
main.py — Entry point for training.

Usage:
    python main.py                          # defaults from args.py
    python main.py --epochs 30 --lr 0.0005 # custom hyperparams
    python main.py --backbone fasterrcnn_mobilenet_v3
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from args import get_args, DEVICE
from augmentations import build_train_transforms, build_val_transforms
from dataset import ObjDetectionDataset
from model import build_model
from trainer import train_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def plot_metrics(metrics: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(metrics["epoch"], metrics["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics["epoch"], metrics["val_score"], label="Val Score", marker="o", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Detection Accuracy (%)")
    axes[1].set_title("Validation Detection Score  (IoU ≥ 0.5)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = out_path / "training_metrics.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Metrics plot saved → {plot_file}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    print(f"\n{'='*50}")
    print(f"  Device    : {DEVICE.upper()}")
    print(f"  Backbone  : {args.backbone}")
    print(f"  Image size: {args.image_size}px")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs    : {args.epochs}")
    print(f"{'='*50}\n")

    # ── 1. Load CSVs ──────────────────────────────────────────────────────────
    csv_dir  = Path(args.csv_dir)
    train_csv = csv_dir / "train_data.csv"
    val_csv   = csv_dir / "val_data.csv"

    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\nRun  python prepare_data.py  first."
            )

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}\n")

    # ── 2. Datasets & DataLoaders ─────────────────────────────────────────────
    train_dataset = ObjDetectionDataset(train_df, build_train_transforms(args.image_size))
    val_dataset   = ObjDetectionDataset(val_df,   build_val_transforms(args.image_size))

    # num_workers=0 is safest on macOS with multiprocessing
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False,
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = build_model(args.backbone)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = train_model(model, train_loader, val_loader, args, out_dir)

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    plot_metrics(metrics, out_dir)


if __name__ == "__main__":
    main()
