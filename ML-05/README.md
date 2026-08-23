# LAB05 - Support Vector Machine (SVM)

งานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เพื่อศึกษาและประยุกต์ใช้อัลกอริทึม Support Vector Machine (SVM) ในการจำแนกรูปภาพท่าทางมือเป่ายิ้งฉุบ (Rock-Paper-Scissors Image Classification) และเปรียบเทียบประสิทธิภาพของ 3 Kernels (Linear, Polynomial, RBF)

## รายละเอียดไฟล์ในโฟลเดอร์

* **data/**: โฟลเดอร์เก็บชุดข้อมูลรูปภาพ 3 คลาส (`paper`, `rock`, `scissors`)
* **data_load.py**: สคริปต์โหลดรูปภาพ ตรวจจับโฟลเดอร์คลาส และแปลงเป็นภาพขาวดำ
* **preprocess.py**: สคริปต์ปรับขนาดภาพเป็น 100x100 และแปลงเป็น Feature Vector 10,000 มิติ
* **split_data.py**: สคริปต์แบ่งข้อมูล Train (80%) และ Test (20%) แบบ Stratified
* **svm_model.py**: สคริปต์ Pipeline รวม StandardScaler + PCA (150 Components) และฟังก์ชันสร้างโมเดล SVM 3 Kernels
* **evaluate.py**: สคริปต์ประเมินผลความแม่นยำและบันทึกภาพ Confusion Matrix
* **main.py**: สคริปต์หลักสำหรับรันเปรียบเทียบ Linear, Polynomial, RBF และบันทึกโมเดลที่ดีที่สุด
* **test_svm.py**: สคริปต์สุ่มรูปภาพ 4 รูปมาทดสอบทำนายผลลัพธ์และแสดงภาพผลลัพธ์

## การทำงานของระบบ

1. **Image Preprocessing & PCA**: โหลดภาพมือ ย่อขนาดเป็น $100 \times 100$ พิกเซล และลดมิติข้อมูลด้วย PCA จาก 10,000 มิติเหลือ 150 มิติ เพื่อเพิ่มความเร็วในการเทรน
2. **Model Training (3 Kernels)**: เทรนโมเดล SVM ทั้ง 3 รูปแบบ ได้แก่ Linear Kernel, Polynomial Kernel (Degree=3), และ RBF Kernel
3. **Model Evaluation & Testing**: ประเมินผลเปรียบเทียบค่า Accuracy Score, Confusion Matrix ของแต่ละ Kernel และสุ่มรูปภาพทดสอบทำนายผล

## ผลการประเมินประสิทธิภาพ (Model Evaluation)

| SVM Kernel | Accuracy (%) |
| :--- | :--- |
| **Linear Kernel** | 90.45% |
| **Polynomial Kernel (Degree=3)** | 88.64% |
| **RBF Kernel (Best Kernel)** | **92.27%** |

* **The Best Kernel**: **RBF Kernel** ให้ค่าความแม่นยำสูงสุดที่ **92.27%** เนื่องจากสามารถสร้าง Hyperplane แบ่งเขตการตัดสินใจในรูปทรงที่ไม่เป็นเส้นตรง (Non-linear) ของรูปภาพได้ดีที่สุด

## แหล่งที่มาของข้อมูล (Data Credit)

* **Dataset**: Rock-Paper-Scissors Images Dataset
* **ผู้สร้าง/รวบรวมข้อมูล**: Dr. Julien de la Bruère-Terreault
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
