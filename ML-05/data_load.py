import os
import cv2
import numpy as np
from preprocess import preprocess_image

# กำหนดนามสกุลไฟล์ภาพที่รองรับ
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path="data", img_size=100, max_per_class=None):
    """
    โหลดข้อมูลรูปภาพจากโฟลเดอร์ตามโครงสร้างคลาส:
    - ตรวจจับชื่อโฟลเดอร์ย่อยใน data/ เพื่อใช้เป็นชื่อคลาสโดยอัตโนมัติ
    - อ่านไฟล์ภาพเข้ามา แปลงภาพด้วย preprocess_image
    - คืนค่าภาพทั้งหมด (X), เลเบลตัวเลข (y) และรายชื่อคลาส (classes)
    """
    images = []
    labels = []

    # ดึงรายชื่อโฟลเดอร์ย่อย (paper, rock, scissors) และเรียงตามลำดับอักษร
    # ละเว้นโฟลเดอร์ที่ขึ้นต้นด้วยจุด (.) เช่น ไฟล์ระบบซ่อน
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder)) and not folder.startswith('.')
    ])
    print("Detected classes:", classes)

    # วนลูปอ่านข้อมูลรูปภาพในแต่ละคลาส
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        
        # กรองเอาเฉพาะไฟล์ที่มีนามสกุลภาพที่กำหนด
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            # ตรวจสอบขีดจำกัดจำนวนภาพสูงสุดต่อคลาส (ป้องกันการใช้ RAM มากเกินไป)
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # นำภาพไปปรับขนาดและแปลงเป็น Grayscale
            image = preprocess_image(image, img_size)
            
            # หากภาพเสียหรืออ่านไม่ได้ ให้ข้ามไป
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # ซ้อนภาพทั้งหมดให้เป็น NumPy Array ก้อนเดียว (N, 100, 100)
    return np.stack(images), np.array(labels), classes
