import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    """
    ฟังก์ชันปรับแต่งรูปภาพเบื้องต้นทีละรูป:
    1. ตรวจสอบความสมบูรณ์ของไฟล์ภาพ
    2. แปลงภาพสี (BGR) ให้เป็นภาพขาวดำ (Grayscale) เพื่อลดความซับซ้อนของข้อมูล
    3. ปรับขนาดภาพ (Resize) ให้เป็นขนาดมาตรฐาน 100x100 พิกเซล
    """
    # ตรวจสอบว่าภาพว่างเปล่าหรือเสียหายหรือไม่ หากอ่านไม่ได้ให้คืนค่า None
    if image is None or image.size == 0:
        return None

    # หากภาพมี 3 แชนเนล (ภาพสี) ให้แปลงเป็น Grayscale (เหลือ 1 แชนเนล)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ปรับขนาดภาพเป็น img_size x img_size (100x100 พิกเซล)
    # ใช้ INTER_AREA ซึ่งเป็นอัลกอริทึมที่เหมาะที่สุดสำหรับการย่อขนาดรูปภาพ
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )
    return image

def to_features(images):
    """
    แปลงอาเรย์รูปภาพ 2 มิติ ให้กลายเป็น Feature Vector 1 มิติ สำหรับโมเดล SVM:
    - แปลงจากรูปขนาด (N, 100, 100) -> เวกเตอร์ 1 มิติ ขนาด (N, 10000)
    - ปรับช่วงค่าความสว่างพิกเซล (Normalization) จาก 0-255 ให้กลายเป็น 0.0-1.0 (float32)
    """
    # คลี่รูปภาพแต่ละภาพให้เป็นเวกเตอร์แนวยาว 10,000 มิติ (100 x 100 = 10,000)
    features = images.reshape(len(images), -1).astype(np.float32)
    
    # ทำ Normalization หารด้วย 255.0 เพื่อช่วยให้แบบจำลองคำนวณและลู่เข้าได้รวดเร็วขึ้น
    features /= 255.0
    return features
