"""
轻量版：仅用 OpenCV Haar（笑脸级联）判断微笑；伤心用嘴部几何启发式。
不依赖 TensorFlow，安装更快：pip install opencv-python numpy
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
MIRROR = True

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def classify(face_gray: np.ndarray) -> tuple[str, float]:
    smiles = smile_cascade.detectMultiScale(
        face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(40, 40)
    )
    if len(smiles) > 0:
        return "Smile", min(95.0, 55.0 + 8.0 * len(smiles))

    # 简单“伤心”启发：下半脸边缘能量偏低且无不到笑脸 → sad-ish
    h, w = face_gray.shape
    mouth = face_gray[int(h * 0.55) : int(h * 0.90), int(w * 0.2) : int(w * 0.8)]
    if mouth.size == 0:
        return "Neutral", 40.0
    edges = cv2.Canny(mouth, 50, 120)
    density = float(np.count_nonzero(edges)) / edges.size
    if density < 0.045:
        return "Sad", 60.0
    return "Neutral", 50.0


def main() -> None:
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    cap = None
    for be in backends:
        cap = cv2.VideoCapture(CAMERA_INDEX, be)
        if cap.isOpened():
            break
        cap.release()
    if cap is None or not cap.isOpened():
        raise SystemExit("Cannot open camera")

    Path("captures").mkdir(exist_ok=True)
    win = "Emotion Cam Lite"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
        label, conf = "No face", 0.0
        for (x, y, w, h) in faces[:1]:
            roi = gray[y : y + h, x : x + w]
            label, conf = classify(roi)
            color = (80, 220, 100) if label == "Smile" else (80, 80, 255) if label == "Sad" else (200, 200, 200)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {conf:.0f}%", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, "q quit | s shot", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            p = Path("captures") / f"lite_{int(time.time())}.jpg"
            cv2.imwrite(str(p), frame)
            print("saved", p)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
