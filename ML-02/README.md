# LAB02 - Data Preprocessing

ใบงานนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (ML) เกี่ยวกับการทำ Data Preprocessing (การเตรียมข้อมูล) โดยใช้ชุดข้อมูลประวัติราคาหุ้นของบริษัท Microsoft (MSFT) 

## รายละเอียดไฟล์ในโฟลเดอร์
* **Lab2_pantakan.ipynb**: ไฟล์ Source Code (Jupyter Notebook) ที่ใช้ในการทำ Data Exploration, Visualization, Data Cleaning และ Feature Engineering
* **Microsoft_stock_history.csv**: ชุดข้อมูลดิบราคาหุ้น Microsoft (ช่วงปี 1986 - 2026)

## การทำงานของระบบ
1. **Data Exploration & Visualization**: ตรวจสอบโครงสร้างข้อมูลเบื้องต้น พล็อตกราฟ Histogram ดูการกระจายตัวของราคาหุ้น และทำ Heatmap เพื่อดูค่า Correlation ของแต่ละฟีเจอร์
2. **Data Cleaning**: จัดการแปลงชนิดข้อมูลคอลัมน์ Date, ตรวจสอบและจัดการค่าสูญหาย (Null Values) และข้อมูลซ้ำ (Duplicates)
3. **Feature Engineering**: ทำการแปลงข้อมูลประเภทข้อความ (Categorical Data) ให้อยู่ในรูปตัวเลขด้วยวิธี Label Encoding และ One-Hot Encoding เพื่อเตรียมพร้อมสำหรับนำไปเทรนโมเนล AI

## แหล่งที่มาของข้อมูล (Data Credit)
* **Dataset**: Microsoft Stock Details - Updated Regularly
* **ผู้สร้าง/รวบรวมข้อมูล**: Kalilur Rahman (Dataset Creator)
* **แพลตฟอร์ม**: Kaggle
* **ลิงก์ข้อมูล**: https://www.kaggle.com/datasets/kalilurrahman/microsoft-stock-details-updated-regularly