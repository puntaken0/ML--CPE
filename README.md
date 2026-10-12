# LAB07 - Convolutional Neural Network (CNN) and Applications

โครงงานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เพื่อศึกษาและประยุกต์ใช้โมเดล Convolutional Neural Network (CNN) ในการจำแนกรูปภาพท่าทางมือเป่ายิ้งฉุบ (Rock-Paper-Scissors) พร้อมเปรียบเทียบประสิทธิภาพระหว่างสถาปัตยกรรม CNN (Configurations) และจำนวนรอบการเรียนรู้ (Epochs)

## รายละเอียดไฟล์ในโครงงาน

* **data/**: โฟลเดอร์เก็บรูปภาพ 3 คลาส (`paper`, `rock`, `scissors`)
* **data_loader.py**: สคริปต์โหลดรูปภาพและปรับขนาดมาตรฐาน (100x100 RGB)
* **preprocessing.py**: สคริปต์จัดรูปแบบอาร์เรย์ภาพ 3 มิติให้อยู่ในรูปแบบ uint8
* **split_data.py**: สคริปต์แบ่งข้อมูล Train (70%), Validation (10%), และ Test (20%) แบบ Stratified
* **cnn_model.py**: สคริปต์สร้างโครงสร้าง CNN (Simple, Medium, Deep) พร้อมกลไก EarlyStopping และ ReduceLROnPlateau
* **evaluate.py**: สคริปต์ประเมินผล สร้าง Confusion Matrix และกราฟสรุป Training/Validation Curves
* **main.py**: สคริปต์หลักสำหรับรันเปรียบเทียบ Configurations และ Epochs พร้อมเซฟโมเดลที่ดีที่สุด
* **test_cnn.py**: สคริปต์สุ่มรูปภาพ 4 รูปมาทดสอบทำนายผล พร้อมแสดงค่าความมั่นใจ (Confidence Score %)

## ผลการประเมินประสิทธิภาพ (Model Evaluation)

### 1. เปรียบเทียบโครงสร้าง CNN (Configurations)

| Configuration | รายละเอียดสถาปัตยกรรม | Accuracy (%) |
| :--- | :--- | :--- |
| **Simple (1 Block)** | Conv2D(32) + MaxPool + Dense(64) | 88.57% |
| **Medium (2 Blocks)** | Conv2D(32) + Conv2D(64) + MaxPool + Dense(128) | 94.76% |
| **Deep (3 Blocks)** | Conv2D(32, 64, 128) + BatchNorm + MaxPool + Dropout | **98.33%** |

### 2. เปรียบเทียบจำนวนรอบการเรียนรู้ (Epochs) บน Deep Model

| จำนวน Epochs | Accuracy (%) | ข้อสังเกต |
| :--- | :--- | :--- |
| **10 Epochs** | 92.86% | โมเดลเริ่มสกัดฟีเจอร์ขอบและมุมได้ดี แต่ยังลู่เข้าไม่เต็มที่ |
| **20 Epochs** | 97.14% | ค่า Loss ลดลงอย่างมีเสถียรภาพ และความแม่นยำสูงขึ้นมาก |
| **30 Epochs (Best)** | **98.33%** | โมเดลคืนค่าน้ำหนักที่ดีที่สุดโดยมี EarlyStopping ควบคุมไม่ให้ Overfitting |

## แหล่งที่มาของข้อมูล (Data Credit)

* **Dataset**: Rock-Paper-Scissors Images Dataset
* **ผู้สร้าง/รวบรวมข้อมูล**: Dr. Julien de la Bruère-Terreault
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
