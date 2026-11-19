import cv2
import numpy as np

def preprocess_image(img_path):
 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 img = cv2.GaussianBlur(img, (5,5), 0)
 _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
 return thresh