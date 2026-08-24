# LAB06 - Neural Network and Applications

งานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เพื่อศึกษาและประยุกต์ใช้โมเดล Artificial Neural Network (Multilayer Perceptron: MLP) ในการจำแนกรูปภาพท่าทางมือเป่ายิ้งฉุบ (Rock-Paper-Scissors) พร้อมเปรียบเทียบผลลัพธ์ตามขนาดสถาปัตยกรรม (Configurations) และจำนวนรอบการเรียนรู้ (Epochs)

## รายละเอียดไฟล์ในโฟลเดอร์

* **data/**: โฟลเดอร์เก็บชุดข้อมูลรูปภาพ 3 คลาส (`paper`, `rock`, `scissors`)
* **data_loader.py**: สคริปต์โหลดรูปภาพและแปลงเป็นสี RGB (100x100)
* **preprocessing.py**: สคริปต์เตรียมข้อมูลและจัดรูปแบบอาร์เรย์ให้อยู่ในรูปแบบ uint8
* **split_data.py**: สคริปต์แบ่งข้อมูล Train (70%), Validation (10%), และ Test (20%) แบบ Stratified
* **nn_model.py**: สคริปต์สร้างและเทรนโครงข่าย Neural Network (Simple, Medium, Deep) พร้อมกลไก EarlyStopping และ ReduceLROnPlateau
* **evaluate.py**: สคริปต์ประเมินผล สร้าง Confusion Matrix และวาดกราฟ Training/Validation Loss & Accuracy Curves
* **main.py**: สคริปต์หลักสำหรับรันเปรียบเทียบ Configurations และ Epochs พร้อมบันทึกโมเดลที่ดีที่สุด
* **test_nn.py**: สคริปต์สุ่มรูปภาพ 4 รูปมาทำนายผล แสดงค่าความมั่นใจ (Confidence Score %) และบันทึกรูปผลลัพธ์

## ผลการประเมินประสิทธิภาพ (Model Evaluation)

### 1. เปรียบเทียบโครงสร้าง Neural Network (Configurations)

| Configuration | รายละเอียดสถาปัตยกรรม | Accuracy (%) |
| :--- | :--- | :--- |
| **Simple (1 Layer)** | Dense(64) + Dropout(0.2) | 78.41% |
| **Medium (2 Layers)** | Dense(128) + Dense(64) + BatchNorm + Dropout | 86.82% |
| **Deep (3 Layers)** | Dense(256) + Dense(128) + Dense(64) + BatchNorm + Dropout | **93.64%** |

### 2. เปรียบเทียบจำนวนรอบการเรียนรู้ (Epochs) บน Deep Model

| จำนวน Epochs | Accuracy (%) | ข้อสังเกต |
| :--- | :--- | :--- |
| **10 Epochs** | 85.23% | โมเดลยังเรียนรู้รูปแบบซับซ้อนของรูปภาพได้ไม่เต็มที่ (Underfitting เล็กน้อย) |
| **20 Epochs** | 91.82% | ค่า Loss ลดลงอย่างต่อเนื่องและเริ่มทรงตัว |
| **30 Epochs (Best)** | **93.64%** | โมเดลทำงานร่วมกับ EarlyStopping คืนค่าน้ำหนักที่ดีที่สุดโดยไม่เกิด Overfitting |

## แหล่งที่มาของข้อมูล (Data Credit)

* **Dataset**: Rock-Paper-Scissors Images Dataset
* **ผู้สร้าง/รวบรวมข้อมูล**: Dr. Julien de la Bruère-Terreault
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
