# Setup Guide

---

## Prerequisites

- Compatible with **macOS**, **Linux**, or **Windows**
- Your dataset: images in one folder, YOLO-format `.txt` labels in another

---

## Step 1 — Create a virtual environment

A virtual environment keeps this project's packages isolated from everything else on your machine.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your prompt — this means it's active.  
> Run `deactivate` any time to leave the environment.

---



## Step 2 — Install Python 3.11

### macOS
# 1. Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.11
brew install python@3.11

# 3. Add it to your PATH (Apple Silicon — M1/M2/M3)
echo 'export PATH="/opt/homebrew/opt/python@3.11/libexec/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile

# 3. Add it to your PATH (Intel Mac)
echo 'export PATH="/usr/local/opt/python@3.11/libexec/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile

# 4. Verify — should print Python 3.11.x
python3 --version

### Linux (Ubuntu)

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11

# Verify
python3.11 --version

---



## Step 3 — Install PyTorch

PyTorch needs a platform-specific install command before everything else.  
**Copy the single line that matches your machine:**

### macOS (Apple Silicon M1/M2/M3) — uses MPS acceleration
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
> MPS is auto-detected at runtime via `torch.backends.mps` — no extra steps.

### macOS (Intel)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Linux / Windows — NVIDIA GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Linux / Windows — CPU only (no GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 4 — Install all other dependencies

in vscode open a folder where you wan to store the project and open the terminal type:

```bash
git clone https://github.com/Fingaerwen/AiReckoRemake.git
```

---


## Step 5 — Install all other dependencies

```bash
Cd into your project aka cd SchoolFolder/AoReckoRemake
pip install -r requirements.txt
```

Verify everything installed correctly:
```bash
python -c "import torch, torchvision, pandas, sklearn, matplotlib; print('All good! Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

---

## Step 6 — Add your data

Create and place your files like this !!! images and labels are NOT TO BE CAPITALIZED IT NEEDS TO BE EXACTLY images and labels !!!:

```
data/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/
    ├── img001.txt
    ├── img002.txt
    └── ...
```

## Step 7 — Prepare the dataset

Run this file **once**. It scans your data folders, pairs images with labels, and creates the train/val CSV split.

```bash
prepare_data.py
```

Expected output:
```
Scanning images : .../data/images
Scanning labels : .../data/labels

Dataset CSV  → .../data/CSVs/dataset.csv   (240 pairs)
Train CSV    → .../data/CSVs/train_data.csv  (192 samples)
Val CSV      → .../data/CSVs/val_data.csv   (48 samples)

Done! You can now run:  python main.py
```

---

## Step 8 — Train

Run This file:

```bash
main.py
```

IF you want to change EPOCHS:
1. Open args.py in VSCode from the file explorer on the left
2. Find this line:
parser.add_argument('--epochs', type=int, default=5)

change 5 to however many epochs you want

progress:

```
==================================================
  Device    : CPU          ← or CUDA / MPS
  Backbone  : fasterrcnn_resnet50_fpn
  Image size: 512px
  Batch size: 4
  Epochs    : 50
==================================================

Epoch   1/50 | Train Loss: 1.2341 | Val Loss: 0.9812 | Val Score: 34.2%
  ✓ New best model saved → runs/best_model.pth
Epoch   2/50 | ...
```

When training finishes, a plot is saved to `runs/training_metrics.png`.

---
