"""
args.py — Centralised training hyperparameters.
All paths are relative to this file so the repo works on any machine.
"""

import argparse
from pathlib import Path

import torch

# ── Resolved at import time — no hardcoded drive letters ──────────────────────
ROOT    = Path(__file__).resolve().parent
CSV_DIR = ROOT / "data" / "CSVs"
OUT_DIR = ROOT / "runs"          # saved model checkpoints + plots land here

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="Faster R-CNN training options")

    parser.add_argument(
        "--backbone", type=str, default="fasterrcnn_resnet50_fpn",
        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3"],
        help="Detection backbone to use",
    )
    parser.add_argument(
        "--csv_dir", type=str, default=str(CSV_DIR),
        help="Directory containing train_data.csv and val_data.csv",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(OUT_DIR),
        help="Directory where checkpoints and plots are saved",
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Input image size (square) used by transforms",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size (lower this if you run out of memory)",
    )
    parser.add_argument("--epochs",  type=int,   default=2)
    parser.add_argument("--lr",      type=float, default=0.001,  help="Learning rate")
    parser.add_argument("--wd",      type=float, default=1e-4,   help="Weight decay")

    return parser.parse_args()
