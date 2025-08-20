 # Face Mask Detection

A computer vision system that detects whether individuals are wearing a protective face mask.
Includes model training (transfer learning) and a real-time webcam inference demo.

## Features
- Image classification: with_mask / without_mask
- Transfer learning and fine-tuning
- Evaluation: accuracy, precision, recall, F1, confusion matrix
- Real-time webcam pipeline (OpenCV)
- Reproducible training and evaluation scripts

## Repository Structure
- data/ (not tracked) — train/val/test, class subfolders
- models/ — trained weights and checkpoints
- notebooks/ — exploratory analysis
- src/ — training, evaluation, and real-time scripts
- requirements.txt — pinned dependencies

## Quickstart (Windows/PowerShell)
`powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

