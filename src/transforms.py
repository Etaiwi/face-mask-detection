from __future__ import annotations

from torchvision import transforms
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_train_val_transforms(img_size: int = IMG_SIZE):
    """
    MobileNetV2-style preprocessing.

    - Train: RGB, resize, light augmentation (flip), normalize
    - Val:   RGB, resize, normalize
    """
    train_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tfms, val_tfms


def build_eval_transform(img_size: int = IMG_SIZE):
    """
    Deterministic eval/inference transform:
    RGB -> resize -> tensor -> normalize.
    """
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
