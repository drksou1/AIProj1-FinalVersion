"""
trainer.py — Training and validation loops.
Saves the best model checkpoint automatically.
"""

import torch
import torch.optim as optim
from pathlib import Path
from torchvision.ops import box_iou

from args import DEVICE
from utils import show_batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou_loss(predictions, targets):
    """1 − mean(max IoU per prediction).  Lower is better."""
    total_loss, n_preds = 0.0, 0
    for pred, tgt in zip(predictions, targets):
        if len(pred["boxes"]) == 0 or len(tgt["boxes"]) == 0:
            continue
        ious     = box_iou(pred["boxes"], tgt["boxes"])
        max_ious = ious.max(dim=1)[0]
        total_loss += (1.0 - max_ious).sum().item()
        n_preds    += len(pred["boxes"])
    return total_loss / max(n_preds, 1)


def compute_detection_score(predictions, targets, iou_threshold: float = 0.5):
    """% of predicted boxes with max IoU ≥ threshold against ground truth."""
    total_correct, total_preds = 0, 0
    for pred, tgt in zip(predictions, targets):
        if len(pred["boxes"]) == 0 or len(tgt["boxes"]) == 0:
            continue
        ious     = box_iou(pred["boxes"], tgt["boxes"])
        max_ious = ious.max(dim=1)[0]
        total_correct += (max_ious >= iou_threshold).sum().item()
        total_preds   += len(pred["boxes"])
    return (total_correct / max(total_preds, 1)) * 100.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, args, out_dir: Path):
    device = torch.device(DEVICE)
    model.to(device)

    optimizer     = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float("inf")
    preview_shown = False

    metrics = {"epoch": [], "train_loss": [], "val_loss": [], "val_score": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Show one preview of the first training batch (non-blocking)
            if not preview_shown:
                show_batch(images, targets)
                preview_shown = True

            optimizer.zero_grad()
            loss_dict = model(images, targets)   # Faster R-CNN returns a dict of losses
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss             = running_loss / len(train_loader)
        val_loss, val_score    = validate_model(model, val_loader, device)

        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_score"].append(val_score)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Score: {val_score:.1f}%"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best model saved → {ckpt_path}")

    return metrics


# ── Validation loop ───────────────────────────────────────────────────────────

def validate_model(model, val_loader, device):
    model.eval()
    val_loss, val_score = 0.0, 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            val_loss   += compute_iou_loss(predictions, targets)
            val_score  += compute_detection_score(predictions, targets)

    n = max(len(val_loader), 1)
    return val_loss / n, val_score / n
