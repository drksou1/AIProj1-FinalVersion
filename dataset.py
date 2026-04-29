"""
dataset.py — PyTorch Dataset for YOLO-format object detection data.
Reads image/label paths from a CSV produced by prepare_data.py.
"""

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

import augmentations as aug


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = aug.Compose(transform) if transform else aug.Compose([aug.NoTransform()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image and fix EXIF rotation
        img = Image.open(row["images"]).convert("RGB")
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        image = to_tensor(img)

        # Parse YOLO-format label file
        boxes, labels = [], []
        with open(row["labels"]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)   # 0 is background in Faster R-CNN

        # Handle images with no annotations (empty target)
        if boxes:
            target = {
                "boxes":    torch.tensor(boxes,  dtype=torch.float32),
                "labels":   torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
        else:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }

        image, target = self.transform(image, target)
        return image, target
