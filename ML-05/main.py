import json
import os
import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

# กำหนดตัวแปรตั้งค่าระบบ (Configuration Constants)
DATA_PATH = "data"          # โฟลเดอร์ที่จัดเก็บรูปภาพ paper, rock, scissors
OUTPUT_DIR = "outputs"      # โฟลเดอร์สำหรับบันทึกไฟล์ผลลัพธ์
IMG_SIZE = 100              # ขนาดพิกเซลของภาพ (100x100)
TEST_SIZE = 0.2             # สัดส่วนข้อมูลชุดทดสอบ 20%
MAX_PER_CLASS = 700         # จำกัดรูปสูงสุด 700 ภาพต่อคลาสเพื่อประสิทธิภาพการประมวลผล
KERNELS = ["linear", "poly", "rbf"]  # 3 เคอร์เนลที่นำมาเปรียบเทียบ

def main():
    print("--" * 30)
    print("SVM Image Recognition: Rock vs Paper vs Scissors")
    print("--" * 30)

    # สร้างโฟลเดอร์ outputs หากยังไม่มี
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ขั้นตอนที่ 1: โหลดรูปภาพและตรวจจับคลาส
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    # บันทึกแคชข้อมูลและไฟล์ชื่อคลาส classes.json
    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # ขั้นตอนที่ 2: สกัดฟีเจอร์และปรับสเกลค่าพิกเซล
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # ขั้นตอนที่ 3: แบ่งชุดข้อมูล Train 80% และ Test 20%
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    # บันทึกชุดข้อมูล Train/Test เป็นไฟล์ .npy
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # ขั้นตอนที่ 4 & 5: วนลูปเทรนและเปรียบเทียบเคอร์เนลทั้ง 3 แบบ
    print("\n[Step 4] Training and Comparing SVM Kernels...")
    best_accuracy = 0.0
    best_kernel = None
    best_model = None
    best_scaler = None

    for k in KERNELS:
        print(f"\n--- Testing SVM Kernel: {k.upper()} ---")
        # เทรนโมเดลด้วยเคอร์เนล k พร้อมสร้าง Scaler/PCA
        model, scaler = train_svm(X_train, y_train, kernel=k)
        
        # ทำนายผลบนชุดทดสอบ
        predictions = predict_svm(model, scaler, X_test)
        
        # ประเมินผลและบันทึกรูป Confusion Matrix แยกตามเคอร์เนล
        acc = evaluate_model(
            y_test, predictions, classes,
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{k}.png"
        )

        # เก็บโมเดลที่ให้ค่า Accuracy สูงที่สุด
        if acc > best_accuracy:
            best_accuracy = acc
            best_kernel = k
            best_model = model
            best_scaler = scaler

    # ขั้นตอนที่ 6: บันทึกโมเดลและตัวแปลงข้อมูลที่ดีที่สุด
    print("\n" + "=" * 45)
    print(f"Best SVM Kernel is: {best_kernel.upper()} (Accuracy: {best_accuracy * 100:.2f}%)")
    print("=" * 45)

    # เซฟไฟล์โมเดลที่ดีที่สุดและ Scaler ด้วย joblib
    joblib.dump(best_model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(best_scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print(f"Saved best model ({best_kernel}) to {OUTPUT_DIR}/svm_model.pkl")

if __name__ == "__main__":
    main()
