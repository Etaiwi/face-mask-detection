from __future__ import annotations
import argparse
from pathlib import Path
import sys
from collections import deque
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from models.factory import build_model
from transforms import build_eval_transform
from config import (
    CLASSES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_WEIGHTS_PATH,
)
from utils import device, load_weights

# ---- helpers ----
def get_face_detector() -> cv2.CascadeClassifier:
    """
    Built-in OpenCV Haar cascade (zero extra files). Works best with front-facing, decent light.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(cascade_path)
    if clf.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")
    return clf

def crop_largest_face(frame_bgr: np.ndarray, face_clf: cv2.CascadeClassifier):
    """
    Returns (roi_bgr, (x1,y1,x2,y2)) if a face is found, else (frame_bgr, None).
    Pads the crop a bit so mask edges are included.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_clf.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame_bgr, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

def preprocess_bgr(frame_bgr: np.ndarray, tfm: T.Compose) -> torch.Tensor:
    """
    BGR (OpenCV) -> RGB PIL -> normalized tensor [1,3,H,W]
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    x = tfm(im).unsqueeze(0)
    return x

# ---- argparse ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Real-time webcam inference for face-mask classifier")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Path to model weights (.pt)")
    p.add_argument("--classes", nargs="+", default=CLASSES, help="Class names in training order")
    p.add_argument("--img-size", type=int, default=IMG_SIZE)

    # video source & preview
    p.add_argument("--source", default="0", help="Webcam index ('0','1',...) or video file path")
    p.add_argument("--flip", action="store_true", help="Horizontally mirror the preview")
    p.add_argument("--width", type=int, default=0, help="Capture width hint (0=leave default)")
    p.add_argument("--height", type=int, default=0, help="Capture height hint (0=leave default)")

    # display/logic
    p.add_argument("--conf", type=float, default=0.0, help="Confidence threshold for labeling (0..1)")
    p.add_argument("--smooth", type=int, default=5, help="Temporal smoothing window (frames, 0=off)")
    return p

# ---- main ----
def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # parse source (int for webcam, string path for file)
    src: int | str
    if isinstance(args.source, str) and args.source.isdigit():
        src = int(args.source)
    else:
        src = args.source

    # device/model
    dev = device()
    model = build_model(num_classes=len(args.classes)).to(dev)
    load_weights(model, args.weights, map_location=dev)
    model.eval()

    # transforms & detector
    tfm = build_eval_transform(img_size=args.img_size)
    face_clf = get_face_detector()

    # capture
    cap = cv2.VideoCapture(src)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[error] cannot open source: {args.source}", file=sys.stderr)
        return 1

    class_names = list(args.classes)
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = max(0.0, min(1.0, args.conf))
    window = deque(maxlen=max(0, args.smooth))

    print("Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # detect & crop
            roi, box = crop_largest_face(frame, face_clf)
            x = preprocess_bgr(roi, tfm).to(dev, non_blocking=True)

            # predict
            logits = model(x)                        # [1, C]
            probs  = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()  # [C]

            # temporal smoothing
            if window.maxlen and window.maxlen > 0:
                window.append(probs)
                avg = np.mean(window, axis=0)
                use = avg
            else:
                use = probs

            idx  = int(use.argmax())
            conf = float(use[idx])
            label = class_names[idx] if conf >= threshold else "unknown"

            # draw
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)

            text = f"{label} ({conf:.2f})"
            cv2.putText(frame, text, (12, 32), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Face-mask classifier (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
