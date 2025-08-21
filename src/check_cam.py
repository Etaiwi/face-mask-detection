"""
Webcam probe for Face Mask Detection project.

- Opens default camera (index 0), grabs a single frame, and prints its shape (HxWxC)
- Useful to verify OpenCV/video permissions before building the real-time demo

Usage:
    python src/check_cam.py
"""

import sys
import time
import cv2

def open_camera(index: int = 0):
    """Return an opened VideoCapture or None if opening fails."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)    # for windows, for other OS use cv2.CAP_ANY
    if not cap.isOpened(): return None
    return cap

def read_one_frame(cap: cv2.VideoCapture):
    """Grab exactly one frame; return (ok: bool, frame: np.ndarray|None)."""
    ok, frame = cap.read()
    return ok, frame

def main() -> int:
    cap = open_camera(index=0)
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera.", file=sys.stderr)
        return 1
    
    try:
        # small warm-up so exposure/auto-focus can settle
        time.sleep(0.5) 

        # Read ibe frane
        ok, frame = read_one_frame(cap)
        if not ok or frame is None:
            print("Error: Could not read a frame from the camera.", file=sys.stderr)
            return 1
        
        # Validate and print frame dimentions
        if not hasattr(frame, "shape") or frame.size == 0:
            print("Error: Frame is empty or has no shape attribute.", file=sys.stderr)
            return 1
        
        if frame.ndim == 3:
            h, w, c = frame.shape
            print(f"Captured frame shape: {h}x{w}x{c}")
        elif frame.ndim == 2:
            h, w = frame.shape
            print(f"Captured frame shape: {h}x{w} (grayscale)")
        else:
            print(f"Warning: unxepected frame ndim; shape={getattr(frame, 'shape', None)}", file=sys.stderr)
            return 1
        
        return 0
    finally:
        # Release the camera
        cap.release()

if __name__ == "__main__":
    sys.exit(main())