"""
Emotion Cam — 实时摄像头人脸情绪识别（微笑 / 伤心）
高级方案：DeepFace Emotion + OpenCV 显示 + 时间平滑
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

# ---------- 配置 ----------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ANALYZE_EVERY_N = 5
SMOOTH_WINDOW = 12
MIRROR_DEFAULT = True
CAPTURE_DIR = Path("captures")

LABEL_ZH = {
    "angry": "生气",
    "disgust": "厌恶",
    "fear": "害怕",
    "happy": "微笑",
    "sad": "伤心",
    "surprise": "惊讶",
    "neutral": "平静",
}
FOCUS = ("happy", "sad")


@dataclass
class EmotionState:
    dominant: str = "neutral"
    scores: Optional[Dict[str, float]] = None
    face_box: Optional[Tuple[int, int, int, int]] = None


class TemporalSmoother:
    def __init__(self, window: int = SMOOTH_WINDOW) -> None:
        self._buf: Deque[Dict[str, float]] = deque(maxlen=window)

    def push(self, scores: Dict[str, float]) -> Dict[str, float]:
        self._buf.append(scores)
        keys = {k for s in self._buf for k in s}
        avg = {k: 0.0 for k in keys}
        for s in self._buf:
            for k, v in s.items():
                avg[k] += float(v)
        n = max(len(self._buf), 1)
        return {k: v / n for k, v in avg.items()}


def pick_camera(index: int) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def analyze_frame(bgr: np.ndarray) -> EmotionState:
    try:
        results = DeepFace.analyze(
            img_path=bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
    except Exception:
        return EmotionState()

    result = results[0] if isinstance(results, list) else results
    if not result:
        return EmotionState()

    scores = {k: float(v) for k, v in (result.get("emotion") or {}).items()}
    dominant = str(result.get("dominant_emotion") or "neutral")
    region = result.get("region") or {}
    box = None
    if {"x", "y", "w", "h"} <= set(region.keys()):
        box = (int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"]))
    return EmotionState(dominant=dominant, scores=scores, face_box=box)


def focus_label(scores: Dict[str, float]) -> Tuple[str, float, str]:
    happy = scores.get("happy", 0.0)
    sad = scores.get("sad", 0.0)
    if happy >= sad and happy >= 25.0:
        return "Smile / Wei Xiao", happy, "happy"
    if sad > happy and sad >= 25.0:
        return "Sad / Shang Xin", sad, "sad"
    dom = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "neutral"
    # OpenCV putText 对中文支持差，HUD 用拼音/英文；右侧条用英文 key
    en = {
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "sad": "Sad",
        "surprise": "Surprise",
        "neutral": "Neutral",
    }
    return en.get(dom, dom), scores.get(dom, 0.0), dom


def draw_ui(frame: np.ndarray, state: EmotionState, fps: float, mirror: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    scores = state.scores or {}
    title, conf, key = focus_label(scores) if scores else ("Looking for face...", 0.0, "neutral")
    color = (80, 220, 100) if key == "happy" else (80, 80, 255) if key == "sad" else (220, 220, 220)

    cv2.putText(frame, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"conf {conf:.1f}% | FPS {fps:.1f} | mirror {'on' if mirror else 'off'}",
        (24, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    if state.face_box:
        x, y, bw, bh = state.face_box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    if scores:
        bar_x = w - 280
        y0 = 140
        for i, (name, val) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])):
            yy = y0 + i * 36
            label = LABEL_ZH.get(name, name)
            # 用英文避免缺字，旁边括号中文拼音简写
            en = name
            bar_w = int(220 * min(val, 100.0) / 100.0)
            bar_color = (80, 220, 100) if name == "happy" else (80, 80, 255) if name == "sad" else (180, 180, 180)
            thick = 2 if name in FOCUS else 1
            cv2.rectangle(frame, (bar_x, yy), (bar_x + 220, yy + 18), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, yy), (bar_x + bar_w, yy + 18), bar_color, -1)
            cv2.putText(
                frame,
                f"{en} {val:.0f}%",
                (bar_x, yy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                thick,
                cv2.LINE_AA,
            )
            _ = label  # 保留中文映射供扩展 PIL 绘制

    cv2.putText(
        frame,
        "q/Esc quit | s screenshot | f flip",
        (24, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    print("Opening camera...")
    cap = pick_camera(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index={CAMERA_INDEX}")

    print("Warming up DeepFace emotion model (first run downloads weights)...")
    warm = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            img_path=warm,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
    except Exception as e:
        print("warmup note:", e)

    CAPTURE_DIR.mkdir(exist_ok=True)
    smoother = TemporalSmoother()
    state = EmotionState()
    mirror = MIRROR_DEFAULT
    frame_i = 0
    fps = 0.0
    fps_t = time.time()
    fps_n = 0

    win = "Emotion Cam"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame grab failed.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        frame_i += 1
        if frame_i % ANALYZE_EVERY_N == 0:
            raw = analyze_frame(frame)
            if raw.scores:
                smoothed = smoother.push(raw.scores)
                dom = max(smoothed.items(), key=lambda kv: kv[1])[0]
                state = EmotionState(dominant=dom, scores=smoothed, face_box=raw.face_box)
            else:
                state = EmotionState(scores=state.scores, face_box=None)

        fps_n += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_n / (time.time() - fps_t)
            fps_t = time.time()
            fps_n = 0

        shown = draw_ui(frame, state, fps, mirror)
        cv2.imshow(win, shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            mirror = not mirror
        if key == ord("s"):
            path = CAPTURE_DIR / f"shot_{int(time.time())}.jpg"
            cv2.imwrite(str(path), shown)
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()
    print("bye")


if __name__ == "__main__":
    main()
