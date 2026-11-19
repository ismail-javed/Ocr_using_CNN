import cv2
import numpy as np

def segment_characters(binary_img):
    # Find contours
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    # Filter valid contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Better filtering
        if w > 5 and h > 10:
            boxes.append((x, y, w, h))

    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    prev_x_end = None

    for (x, y, w, h) in boxes:
        # ---- SPACE DETECTION ----
        if prev_x_end is not None:
            gap = x - prev_x_end

            # Dynamic threshold based on character width
            if gap > w * 1.5:
                chars.append(None)   # <-- SPACE marker (NOT string)

        # Extract character
        char_img = binary_img[y:y+h, x:x+w]

        # Add padding (important for CNN)
        pad = 4
        char_img = cv2.copyMakeBorder(
            char_img, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0
        )

        # Resize
        char_img = cv2.resize(char_img, (28, 28))

        chars.append(char_img)

        prev_x_end = x + w

    return chars