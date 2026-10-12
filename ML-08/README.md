# LAB08 - Deep Convolutional Neural Network (DCNN) and Applications

โครงงานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เพื่อศึกษาและประยุกต์ใช้โมเดล Deep Convolutional Neural Network (DCNN) ที่มีความลึกของโครงสร้างหลายชั้นในการจำแนกรูปภาพท่าทางมือเป่ายิ้งฉุบ (Rock-Paper-Scissors) พร้อมเปรียบเทียบประสิทธิภาพระหว่างสถาปัตยกรรม DCNN (Configurations) และจำนวนรอบการเรียนรู้ (Epochs)

## รายละเอียดไฟล์ในโครงงาน

* **data/**: โฟลเดอร์รูปภาพ 3 คลาส (`paper`, `rock`, `scissors`)
* **data_loader.py**: สคริปต์โหลดรูปภาพ พร้อมฟังก์ชัน `imread_unicode` รองรับโฟลเดอร์ภาษาไทย
* **preprocessing.py**: สคริปต์ปรับขนาดภาพเป็น 100x100 RGB
* **split_data.py**: สคริปต์แบ่งข้อมูล Train (70%), Validation (10%), และ Test (20%) แบบ Stratified
* **dcnn_model.py**: สคริปต์สร้างโครงสร้าง Deep CNN (3-Blocks, 4-Blocks, DeepDouble) พร้อม Batch Normalization และ Dropout
* **evaluate.py**: สคริปต์ประเมินผล สร้าง Confusion Matrix และกราฟ Training/Validation Curves
* **main.py**: สคริปต์หลักสำหรับรันเปรียบเทียบ Configurations และ Epochs
* **test_dcnn.py**: สคริปต์สุ่มรูปภาพ 4 รูปมาทดสอบทำนายผลพร้อมแสดงค่า Confidence Score

## ผลการประเมินประสิทธิภาพ (Model Evaluation)

### 1. เปรียบเทียบโครงสร้าง DCNN (Configurations)

| Configuration | รายละเอียดสถาปัตยกรรม | Accuracy (%) |
| :--- | :--- | :--- |
| **DCNN-3Blocks** | Conv2D (32, 64, 128) + BatchNorm + MaxPool + Dropout | 94.29% |
| **DCNN-4Blocks** | Conv2D (32, 64, 128, 256) + BatchNorm + MaxPool + Dropout | **98.81%** |
| **DCNN-DeepDouble** | Double-Conv (32x2, 64x2, 128x2) + GlobalAveragePooling | 97.62% |

### 2. เปรียบเทียบจำนวนรอบการเรียนรู้ (Epochs) บนโมเดล DCNN-4Blocks

| จำนวน Epochs | Accuracy (%) | ข้อสังเกต |
| :--- | :--- | :--- |
| **10 Epochs** | 94.52% | โมเดลเริ่มเรียนรู้ลักษณะขอบและพื้นผิวของมือได้อย่างรวดเร็ว |
| **20 Epochs** | 98.10% | ค่า Loss ลดลงอย่างต่อเนื่องและมีความแม่นยำสูงมาก |
| **30 Epochs (Best)** | **98.81%** | โมเดลเรียนรู้ฟีเจอร์ระดับลึกได้อย่างสมบูรณ์ โดยไม่เกิด Overfitting จาก BatchNorm |

## แหล่งที่มาของข้อมูล (Data Credit)

* **Dataset**: Rock-Paper-Scissors Images Dataset
* **ผู้สร้าง/รวบรวมข้อมูล**: Dr. Julien de la Bruère-Terreault
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors