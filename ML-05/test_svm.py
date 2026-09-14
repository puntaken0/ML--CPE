import json
import joblib
import matplotlib
# ใช้ 'Agg' สำหรับประมวลผลกราฟิกโดยไม่ต้องแสดงผลบนหน้าต่าง
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100
N_SAMPLES = 4  # จำนวนรูปภาพที่ต้องการสุ่มนำมาทดสอบ

def test_svm(n_samples=N_SAMPLES):
    """
    โหลดโมเดลที่ดีที่สุดมาทดสอบสุ่มภาพจาก Test Set จำนวน 4 รูป:
    - แสดงภาพตัวอย่างแบบ 2x2 Grid
    - แสดงข้อความคลาสจริง (True) และคลาสที่ทาย (Pred)
    - สีเขียว = ทายถูก, สีแดง = ทายผิด
    - บันทึกผลลัพธ์เป็นไฟล์ prediction_sample.png
    """
    # โหลดโมเดล, ตัวแปลงสเกล/PCA, ชุดข้อมูลทดสอบ และชื่อคลาส
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # สุ่มเลือก Index ของรูปภาพใน X_test มาจำนวน n_samples (ไม่ซ้ำกัน)
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # ทำนายผลโดยแปลงข้อมูลด้วย scaler ก่อนส่งเข้าโมเดล
    predictions = model.predict(scaler.transform(X_sample))

    # จัดเลย์เอาต์การแสดงผลเป็นตาราง 2x2
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    # วาดรูปภาพและข้อความกำกับแต่ละช่อง
    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"

        # คลี่เวกเตอร์ 10,000 มิติกลับมาเป็นภาพ 100x100 เพื่อแสดงผล
        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color)

        print(f"[{i + 1}] Pred: {pred:<10} True: {true:<10} {'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    # บันทึกรูปผลลัพธ์เป็นไฟล์ PNG
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    test_svm()
