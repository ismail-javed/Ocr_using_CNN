import cv2
import numpy as np
import tensorflow as tf
from preprocess import preprocess_image
from segment import segment_characters

# Load model
model = tf.keras.models.load_model("model.h5")

# Correct EMNIST labels
labels = [
'0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J',
'K','L','M','N','O','P','Q','R','S','T',
'U','V','W','X','Y','Z',
'a','b','d','e','f','g','h','n','q','r','t'
]

# Image path
image_path = r"C:\Users\md maskur alam\Desktop\Image_OCR_Module\image.jpg"

# Preprocess
binary = preprocess_image(image_path)

# Segment characters
chars = segment_characters(binary)

result_text = ""

print("Total characters detected:", len(chars))

for i, ch in enumerate(chars):
    # Resize to model input
    ch = cv2.resize(ch, (28, 28))

    # Normalize
    ch = ch.astype("float32") / 255.0

    # 🔥 FIX: EMNIST orientation correction
    ch = np.rot90(ch)
    ch = np.fliplr(ch)

    # Reshape
    ch = ch.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(ch, verbose=0)
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    print(f"Char {i}: {labels[class_id]} ({confidence:.2f})")

    result_text += labels[class_id]

print("\nRecognized Text:", result_text)

# Save output
with open("output.txt", "w") as f:
    f.write(result_text)

# Show processed image
cv2.imshow("Binary Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Output saved to output.txt")