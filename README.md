# Face Mask Detection — Computer Vision Demo

A lightweight computer vision application for detecting whether a person is wearing a face mask, built with PyTorch and MobileNetV2 and deployed as an interactive web demo.

🔗 **Live Demo (Google Cloud Run):**  
https://face-mask-detection-385217468790.us-central1.run.app

---

## Overview

This project implements an end-to-end image classification pipeline:
- training a convolutional neural network for face mask detection,
- evaluating it on held-out data,
- and deploying it as a real-time, user-facing application.

The demo allows users to upload an image or capture one from their camera and receive an immediate prediction.

---

## Model & Approach

- **Architecture:** MobileNetV2 (transfer learning)
- **Framework:** PyTorch
- **Input:** RGB face images, resized to 224×224
- **Classes:** `with_mask`, `without_mask`
- **Preprocessing:** ImageNet normalization
- **Deployment:** Streamlit app running on Google Cloud Run

Model weights are **automatically downloaded** from GitHub Releases on first run if not present locally.

---

## Repository Structure

```
.
├── src/
│   ├── app.py              # Streamlit demo app
│   ├── train.py            # Training loop
│   ├── test.py             # Evaluation script
│   ├── prepare_data.py     # Dataset cleaning & splitting
│   ├── webcam.py           # Local webcam inference
│   ├── model_factory.py    # Model construction
│   ├── transforms.py       # Image transforms
│   └── weights.py          # Auto-download model weights
├── models/
│   └── best_mobilenetv2.pt # Downloaded at runtime (not committed)
├── Dockerfile
├── requirements.txt
└── README.md
```

> The dataset is intentionally **not included** in the repository.

**Expected dataset layout (for training only):**

```
data/
└── raw/
    ├── with_mask/
    └── without_mask/
```

Training scripts (e.g., `src/prepare_data.py`, `src/train.py`) assume this structure.

---

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
```

On first run, the model weights will be downloaded automatically.

---

## Notes & Limitations

- The model performs well on benchmark data but may be sensitive to real-world variations (lighting, occlusion, camera angle).
- This project focuses on **inference and deployment**, not face detection or tracking.
- The Cloud Run service uses CPU-only inference for cost efficiency.

---

## Author

**Etai Wigman**  
Machine Learning · Computer Vision · Software Engineering  
GitHub: https://github.com/Etaiwi
