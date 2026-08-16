# LAB04 - k-Nearest Neighbors (KNN)

งานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เกี่ยวกับการศึกษาและประยุกต์ใช้อัลกอริทึม k-Nearest Neighbors (KNN) ในการจำแนกระดับช่วงราคาสมาร์ตโฟน (Mobile Price Range Classification) โดยใช้ชุดข้อมูลสเปกสมาร์ตโฟน (Mobile Dataset)

## รายละเอียดไฟล์ในโฟลเดอร์

* **Mobile.csv**: ชุดข้อมูลคุณสมบัติและสเปกของสมาร์ตโฟน (2,000 แถว 21 คอลัมน์)
* **data_load.py**: สคริปต์โหลดข้อมูลและกำหนด Class เป้าหมาย (Low cost, Medium cost, High cost, Very High cost)
* **preprocess.py**: สคริปต์ทำความสะอาดข้อมูล ตัดหน่วยข้อความ และแปลงข้อมูลเป็นตัวเลข
* **split_data.py**: สคริปต์แบ่งชุดข้อมูล Train (80%) และ Test (20%) แบบ Stratified
* **knn_model.py**: สคริปต์สร้าง Pipeline ปรับสเกลด้วย StandardScaler และเทรนโมเดล KNN
* **evaluate.py**: สคริปต์ประเมินผลความแม่นยำและสร้างกราฟ Confusion Matrix
* **main.py**: สคริปต์หลักสำหรับรันเปรียบเทียบค่า k และบันทึก Best Model
* **test_knn.py**: สคริปต์สุ่มทดสอบการทำนายผลลัพธ์ของโมเดล

## การทำงานของระบบ

1. **Data Preprocessing**: โหลดข้อมูล `Mobile.csv` จัดการตัดหน่วยข้อความ (เช่น mAh, gb, cm) และแปลงตัวแปร Yes/No ให้อยู่ในรูปตัวเลข พร้อมปรับสเกลข้อมูลด้วย StandardScaler
2. **Model Training & Hyperparameter Tuning**: เทรนโมเดล KNN โดยทดสอบเปรียบเทียบจำนวนเพื่อนบ้านที่แตกต่างกัน ($k = 3, 5, 7, 9, 11$)
3. **Model Evaluation & Testing**: ประเมินผลแต่ละค่า $k$ ด้วย Accuracy Score, Classification Report และ Confusion Matrix พร้อมทำการสุ่มตัวอย่างข้อมูลใน Test Set มาทดสอบทำนายผล

## ผลการประเมินโมเดล (Model Evaluation)

* **k = 3**: Accuracy = 48.00%
* **k = 5**: Accuracy = 47.00%
* **k = 7**: Accuracy = 53.75%
* **k = 9 (Best k)**: Accuracy = 55.75%
* **k = 11**: Accuracy = 55.25%

## แหล่งที่มาของข้อมูล (Data Credit)

* **Dataset**: Mobile Price Classification
* **ผู้สร้าง/รวบรวมข้อมูล**: Abhishek Sharma
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
