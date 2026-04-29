"""
prepare_data.py — Run this ONCE before training.

Expected folder layout (place your data here before running):

    project/
    ├── data/
    │   ├── images/   ← your image files (.jpg, .jpeg, .png, .bmp, .webp)
    │   └── labels/   ← your YOLO-format .txt label files (same stem as images)
    ├── prepare_data.py
    ├── main.py
    └── ...

Outputs (auto-created):
    data/CSVs/dataset.csv
    data/CSVs/train_data.csv   (80 %)
    data/CSVs/val_data.csv     (20 %)
"""

import csv
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
import pandas as pd

# ── Paths (all relative to this file — works on any machine) ──────────────────
ROOT      = Path(__file__).resolve().parent
IMG_DIR   = ROOT / "data" / "images"
LBL_DIR   = ROOT / "data" / "labels"
CSV_DIR   = ROOT / "data" / "CSVs"
CSV_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── 1. Build a stem → path lookup for labels ─────────────────────────────────
def collect_labels(label_dir: Path) -> dict:
    lbl_map = {}
    for f in label_dir.rglob("*.txt"):
        lbl_map[f.stem] = f
    return lbl_map

# ── 2. Pair images with their label files ────────────────────────────────────
def pair_images_labels(img_dir: Path, lbl_map: dict) -> list:
    pairs = []
    missing = []
    for img_path in sorted(img_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = lbl_map.get(img_path.stem)
        if lbl_path:
            pairs.append((str(img_path), str(lbl_path)))
        else:
            missing.append(img_path.name)
    return pairs, missing

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not IMG_DIR.exists():
        sys.exit(f"[ERROR] Image directory not found: {IMG_DIR}\n"
                 "Create it and add your images before running.")
    if not LBL_DIR.exists():
        sys.exit(f"[ERROR] Label directory not found: {LBL_DIR}\n"
                 "Create it and add your YOLO .txt labels before running.")

    print(f"Scanning images : {IMG_DIR}")
    print(f"Scanning labels : {LBL_DIR}")

    lbl_map = collect_labels(LBL_DIR)
    pairs, missing = pair_images_labels(IMG_DIR, lbl_map)

    if missing:
        print(f"\n[WARNING] {len(missing)} image(s) have no matching label and will be skipped:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not pairs:
        sys.exit("[ERROR] No matched image-label pairs found. "
                 "Make sure image stems match label stems (e.g. img001.jpg ↔ img001.txt).")

    # Write full dataset CSV
    dataset_csv = CSV_DIR / "dataset.csv"
    with open(dataset_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["images", "labels"])
        writer.writerows(pairs)
    print(f"\nDataset CSV  → {dataset_csv}  ({len(pairs)} pairs)")

    # Train / val split
    df = pd.read_csv(dataset_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(CSV_DIR / "train_data.csv", index=False)
    val_df.to_csv(CSV_DIR / "val_data.csv",   index=False)

    print(f"Train CSV    → {CSV_DIR / 'train_data.csv'}  ({len(train_df)} samples)")
    print(f"Val CSV      → {CSV_DIR / 'val_data.csv'}   ({len(val_df)} samples)")
    print("\nDone! You can now run:  python main.py")

if __name__ == "__main__":
    main()
