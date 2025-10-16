# Face Mask Detection — MobileNetV2 (PyTorch)

Compact baseline classifier for **with_mask / without_mask** built with PyTorch + torchvision.

## What’s included
- `src/train.py` — transfer learning on ImageFolder (MobileNetV2), saves `models/best_mobilenetv2.pt`
- `src/test.py` — two modes:
  - **Evaluate** labeled folders (`--test-dir`): prints **accuracy + macro F1**
  - **Infer** unlabeled images (`--images`): prints `path -> predicted class`

> Note: per-class reports/confusion matrix and webcam streaming are **not** included in this version.

---

## Project layout
```text
src/
  train.py
  test.py
  models/
    __init__.py
    factory.py          # shared model builder (MobileNetV2 + replaced head)
data/
  train/  with_mask/, without_mask/
  val/    with_mask/, without_mask/
models/                   # weights/artifacts (gitignored)
requirements.txt
README.md
```
**ImageFolder expectation:** for labeled splits (`train/`, `val/`, `test/`), each split contains one subfolder per class:
```text
data/val/
  with_mask/
    img001.jpg
    ...
  without_mask/
    img101.jpg
    ...
```
For **unlabeled inference**, place any images under a free-form folder (no labels needed), e.g.:
```text
data/infer/
  pic1.jpg
  vacation/photo2.png
  nested/anything.webp
```
---

## Setup
```bash
python -m venv .venv
# Windows (PowerShell)
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```
> CUDA note (PyTorch wheels): install Torch per your CUDA. Example for CUDA 12.6:
```bash
> pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```
---

## Train
```bash
python src/train.py \
  --train-dir data/train \
  --val-dir data/val \
  --epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --num-workers 0

- Best weights are saved to: `models/best_mobilenetv2.pt`.
- Uses ImageNet preprocessing and light augmentation (horizontal flip).
```
---

## Evaluate (labeled folder)
```bash
python src/test.py \
  --test-dir data/val \
  --weights models/best_mobilenetv2.pt \
  --classes with_mask without_mask \
  --batch-size 32 \
  --num-workers 0

Outputs a summary like:

device: cuda
eval: n=1448 | acc=0.967 | f1_macro=0.967
```

> Important — class order: `--classes` must match the training order (i.e., the subfolder order used by `ImageFolder`). If the dataset’s detected order differs, the script prints a warning.

---

## Results
Final test-set performance (on `data/test`):
```text
eval: n=1452 | acc=0.966 | f1_macro=0.966
```
---

## Infer (unlabeled images)
```bash
python src/test.py \
  --images data/infer \
  --weights models/best_mobilenetv2.pt \
  --classes with_mask without_mask \
  --batch-size 32 \
  --num-workers 0

- Recursively scans images under `data/infer` (jpg/jpeg/png/bmp/webp).
- Prints `path -> predicted_class` per image, e.g.:
```
```text
data/infer/pic1.jpg -> with_mask
data/infer/folder/photo2.png -> without_mask
```
---

## Webcam (real-time inference)

Run the classifier on live webcam/video using OpenCV face detection (Haar cascade).  
Add `opencv-python` to your `requirements.txt`.

```bash
python src/webcam.py \
  --weights models/best_mobilenetv2.pt \
  --classes with_mask without_mask \
  --img-size 224 \
  --source 0 \
  --flip \
  --conf 0.6 \
  --smooth 5
```
---

## Tips & notes

- Windows workers: start with `--num-workers 0`.
- Reproducibility: a seed is set, but full CUDA determinism is not guaranteed across devices.
- PIL warning: if you see “Palette images with Transparency…”, it’s benign; the code converts to RGB. You can silence it by adding:
```python
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency")
```
near the imports in `src/test.py`.

---

## Requirements
```text
Minimal runtime dependencies (kept lean on purpose):

torch
torchvision
numpy
tqdm
Pillow

> If you later add per-class reports with scikit-learn, also install:
> scikit-learn
```
---

## License
MIT (or your preferred license)
