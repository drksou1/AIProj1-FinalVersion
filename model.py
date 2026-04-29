"""
model.py — Builds a Faster R-CNN detection model.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(backbone: str, num_classes: int = 2):
    """
    Build a pretrained Faster R-CNN and replace its head for `num_classes`.

    num_classes includes the background, so for 1 object class → num_classes=2.
    """
    if backbone == "fasterrcnn_resnet50_fpn":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model   = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    elif backbone == "fasterrcnn_mobilenet_v3":
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model   = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    # Replace the classifier head so it outputs `num_classes` categories
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
