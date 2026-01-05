from __future__ import annotations

from torchvision import transforms
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

def to_rgb(img):
    # img is a PIL Image; make sure it's RGB
    return img.convert("RGB")

def build_train_val_transforms(img_size: int = IMG_SIZE):
    train_tfms = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms

def build_eval_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    