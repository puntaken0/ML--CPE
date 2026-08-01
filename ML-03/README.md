# LAB03 - Regression & Classification

งานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เกี่ยวกับการทำ Regression (การทำนายราคาหุ้น) และ Classification (การทายทิศทางหุ้นขึ้น/ลง) โดยใช้ชุดข้อมูลประวัติราคาหุ้นของบริษัท Microsoft (MSFT)

## รายละเอียดไฟล์ในโฟลเดอร์
* **lab_3.ipynb**: ไฟล์ Source Code (Jupyter Notebook) ที่ใช้ในการทำ Regression (Simple/Multiple + PCA) และ Classification (Logistic Regression + PCA)
* **Microsoft_stock_history.csv**: ชุดข้อมูลดิบราคาหุ้น Microsoft

## การทำงานของระบบ
1. **LAB 1 - Regression**: ทำนายราคาปิด (`Close`) โดยใช้ Simple Linear Regression และ Multiple Linear Regression ร่วมกับการลดมิติข้อมูลด้วย PCA
2. **LAB 2 - Classification**: แปลงโจทย์เป็นการทายทิศทางราคาหุ้น (ขึ้น/ลง) โดยใช้ Logistic Regression พร้อมสร้าง Decision Boundary และวัดผลด้วย Confusion Matrix, ROC/AUC
3. **LAB 3 - Model Comparison**: เปรียบเทียบประสิทธิภาพระหว่างโมเดลและประเมินผลการเรียนรู้ของโมเดล (Train vs Test Performance)

## ผลการประเมินโมเดล (Model Evaluation)
* **Simple Linear Regression**: RMSE = 1.70 | R² = 0.9998
* **Multiple Linear Regression (PCA)**: RMSE = 1.16 | R² = 0.9999
* **Logistic Regression (Classification)**: Accuracy = 49.31% | F1-Score = 0.6080

## แหล่งที่มาของข้อมูล (Data Credit)
* **Dataset**: Microsoft Stock Details - Updated Regularly
* **ผู้สร้าง/รวบรวมข้อมูล**: Kalilur Rahman (Dataset Creator)
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/kalilurrahman/microsoft-stock-details-updated-regularly