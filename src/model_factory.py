from __future__ import annotations
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int) -> nn.Module:
    """
    Must match the train-time architecture:
    - MobileNetV2 backbone
    - Replace final classifier layer to have 'out_features = num_classes
    - No pretrained weights here; test-time will load the trained weights
    """
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m