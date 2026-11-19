"""
webcam_ocr_v2.py  —  Live Handwritten Character Recognition (EMNIST Balanced)
Best-practice rewrite: confidence gating, correct label map, inverted contrast,
contour-based multi-char segmentation, FPS counter, and clean resource teardown.

Requirements:
    pip install opencv-python tensorflow numpy

Usage:
    python webcam_ocr_v2.py
    python webcam_ocr_v2.py --model path/to/model.h5 --conf 0.80 --cam 0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EMNIST Balanced — official 47-class label map
#     Digits 0-9, uppercase A-Z, then 11 disambiguated lowercase letters
#     (a b d e f g h n q r t) that don't look like their uppercase form.
#     Source: https://www.nist.gov/system/files/documents/2021/10/15/emnist-matlab.zip
# ─────────────────────────────────────────────────────────────────────────────
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
assert len(LABELS) == 47, "Label list must have exactly 47 entries."

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Model input spec
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 28          # EMNIST image size
IMG_SHAPE  = (1, IMG_SIZE, IMG_SIZE, 1)   # batch=1, H, W, C


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing  (single ROI → model-ready tensor)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(roi: np.ndarray) -> np.ndarray:
    """
    BGR crop → float32 tensor  [1, 28, 28, 1]  ready for CNN inference.

    Steps:
        1. Grayscale
        2. Resize to 28×28 (INTER_AREA is best for downscaling)
        3. Invert: EMNIST expects WHITE character on BLACK background;
           a webcam frame is the opposite.
        4. Normalise to [0, 1]
        5. Add batch + channel dims
    """
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized                      # ← critical inversion step
    normed  = inverted / 255.0
    return normed.reshape(IMG_SHAPE).astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Fast inference  (avoids Keras predict() overhead in tight loops)
# ─────────────────────────────────────────────────────────────────────────────
def predict_char(model: tf.keras.Model, tensor: np.ndarray, conf_thresh: float):
    """
    Returns (char, confidence) or ('?', confidence) if below threshold.
    Calls the model as a callable — much faster than model.predict().
    """
    logits  = model(tensor, training=False)        # shape (1, 47)
    probs   = tf.nn.softmax(logits, axis=-1).numpy()[0]
    idx     = int(np.argmax(probs))
    conf    = float(probs[idx])
    char    = LABELS[idx] if conf >= conf_thresh else "?"
    return char, conf


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-character segmentation inside the ROI
#     (contour-based; works well for isolated printed/handwritten chars)
# ─────────────────────────────────────────────────────────────────────────────
def segment_chars(roi: np.ndarray, min_area: int = 200):
    """
    Find candidate character bounding boxes inside `roi` via contours.
    Returns list of (x, y, w, h) sorted left-to-right.
    Falls back to the whole ROI if nothing useful is found.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Otsu threshold — adaptive to lighting
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Small morphological close to connect near strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and h > 10 and w > 5:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])           # left-to-right order
    return boxes if boxes else [(0, 0, roi.shape[1], roi.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
GREEN      = (0, 220, 80)
RED        = (0, 60, 220)
YELLOW     = (0, 210, 255)
WHITE      = (255, 255, 255)
DARK       = (30, 30, 30)


def draw_roi_overlay(frame, rx1, ry1, rx2, ry2):
    """Semi-transparent darkened border outside the ROI."""
    overlay = frame.copy()
    # Darken everything outside ROI
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], ry1), DARK, -1)
    cv2.rectangle(overlay, (0, ry2), (frame.shape[1], frame.shape[0]), DARK, -1)
    cv2.rectangle(overlay, (0, ry1), (rx1, ry2), DARK, -1)
    cv2.rectangle(overlay, (rx2, ry1), (frame.shape[1], ry2), DARK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    # ROI border
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), GREEN, 2)
    cv2.putText(frame, "ROI", (rx1 + 4, ry1 - 8),
                FONT, 0.55, GREEN, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                FONT, 0.65, WHITE, 1, cv2.LINE_AA)


def draw_prediction_bar(frame, results: list[tuple[str, float]], ry2: int):
    """
    Draw a bottom bar showing predicted characters + confidence bars.
    `results` is [(char, conf), ...]
    """
    bar_y   = ry2 + 12
    x_start = 12
    box_w   = 56
    for i, (char, conf) in enumerate(results):
        bx = x_start + i * (box_w + 8)
        # Background pill
        cv2.rectangle(frame, (bx, bar_y), (bx + box_w, bar_y + 52),
                      (50, 50, 50), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bx, bar_y), (bx + box_w, bar_y + 52),
                      WHITE, 1, cv2.LINE_AA)
        # Character
        color = GREEN if char != "?" else YELLOW
        cv2.putText(frame, char, (bx + 14, bar_y + 32),
                    FONT, 1.0, color, 2, cv2.LINE_AA)
        # Confidence bar
        bar_fill = int(conf * box_w)
        cv2.rectangle(frame, (bx, bar_y + 40), (bx + box_w, bar_y + 50),
                      (80, 80, 80), -1)
        cv2.rectangle(frame, (bx, bar_y + 40), (bx + bar_fill, bar_y + 50),
                      GREEN if conf >= 0.75 else YELLOW, -1)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(model_path: str, conf_thresh: float, cam_index: int):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Loading model from {model_path} …")
    model = tf.keras.models.load_model(str(model_file))
    model.trainable = False
    # Warm-up pass so first real frame isn't slow
    _ = model(np.zeros(IMG_SHAPE, dtype="float32"), training=False)
    print("[INFO] Model loaded. Press 'q' to quit, 'r' to reset ROI size.")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ROI: centred, 200×200
    roi_size = 200

    prev_time  = time.perf_counter()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — retrying …")
            continue

        h, w = frame.shape[:2]
        rx1 = (w - roi_size) // 2
        ry1 = (h - roi_size) // 2
        rx2 = rx1 + roi_size
        ry2 = ry1 + roi_size

        roi = frame[ry1:ry2, rx1:rx2]

        # ── Segmentation + per-char prediction ────────────────────────────
        boxes   = segment_chars(roi)
        results = []
        for (bx, by, bw, bh) in boxes[:10]:   # cap at 10 chars
            char_crop = roi[by : by + bh, bx : bx + bw]
            # Pad to square before resize to preserve aspect ratio
            sz = max(bw, bh)
            pad_h = (sz - bh) // 2
            pad_w = (sz - bw) // 2
            padded = cv2.copyMakeBorder(char_crop, pad_h, pad_h, pad_w, pad_w,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255))
            tensor = preprocess(padded)
            char, conf = predict_char(model, tensor, conf_thresh)
            results.append((char, conf))

            # Draw box around each character inside the ROI view
            cv2.rectangle(frame,
                          (rx1 + bx, ry1 + by),
                          (rx1 + bx + bw, ry1 + by + bh),
                          RED, 1)
            cv2.putText(frame, f"{char} {conf:.0%}",
                        (rx1 + bx, ry1 + by - 4),
                        FONT, 0.45, RED, 1, cv2.LINE_AA)

        # ── Overlays ───────────────────────────────────────────────────────
        draw_roi_overlay(frame, rx1, ry1, rx2, ry2)
        if results:
            draw_prediction_bar(frame, results, ry2)

        # FPS
        now       = time.perf_counter()
        fps_inst  = 1.0 / max(now - prev_time, 1e-9)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst   # exponential smoothing
        prev_time = now
        draw_fps(frame, fps_smooth)

        cv2.imshow("Live Handwritten OCR — EMNIST CNN", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            roi_size = min(roi_size + 20, min(w, h) - 20)
        elif key == ord("-"):
            roi_size = max(roi_size - 20, 60)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live handwritten character recognition — EMNIST CNN")
    parser.add_argument("--model", default="ocr_cnn_tf_model.h5",
                        help="Path to .h5 / SavedModel")
    parser.add_argument("--conf",  type=float, default=0.75,
                        help="Minimum confidence threshold (0–1)")
    parser.add_argument("--cam",   type=int,   default=0,
                        help="Camera device index")
    args = parser.parse_args()
    run(args.model, args.conf, args.cam)