"""
Environment probe for the Face Mask Detection project.

- Prints versions of Python, PyTorch, TorchVision, OpenCV, scikit-learn, and Matplotlib
- Reports CUDA availability, CUDA toolkit build, and GPU name (if present)

Usage:
    python src/check_env.py
"""

import sys
import torch
import torchvision
import cv2
import sklearn
import matplotlib

def print_versions():
    # Python
    print(f"Python version: {sys.version.split()[0]}")

    # Pytorch + TorchVision
    print(f"PyTorch version: {torch.__version__}")
    print(f"TorchVision version: {torchvision.__version__}")

    # OpenCV
    print(f"OpenCV version: {cv2.__version__}")

    # Scikit-learn
    print(f"scikit-learn version: {sklearn.__version__}")

    # Matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")

    # CUDA
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        print(f"CUDA toolkit version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    print_versions()
    sys.exit(0)