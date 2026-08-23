import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    """แปลงภาพเป็น Grayscale และปรับขนาดภาพเป็น 100x100"""
    if image is None or image.size == 0:
        return None

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )
    return image

def to_features(images):
    """คลี่ภาพเป็น Feature Vector 10,000 มิติ และ Normalize ค่าพิกเซลเป็น 0-1"""
    features = images.reshape(len(images), -1).astype(np.float32)
    features /= 255.0
    return features