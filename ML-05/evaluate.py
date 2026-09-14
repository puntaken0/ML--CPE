import matplotlib
# กำหนด Backend เป็น 'Agg' เพื่อให้บันทึกรูปภาพได้โดยไม่ต้องเปิดหน้าต่าง GUI แสดงผล
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_test, predictions, classes, save_path=None):
    """
    ประเมินประสิทธิภาพของแบบจำลอง:
    - คำนวณ Accuracy Score
    - พิมพ์รายงาน Classification Report (Precision, Recall, F1-Score)
    - พิมพ์และบันทึกรูปภาพ Confusion Matrix
    """
    labels = list(range(len(classes)))
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    # หากมีการระบุ save_path ให้วาดกราฟและเซฟเป็นไฟล์รูปภาพ
    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy

def plot_confusion_matrix(matrix, classes, save_path):
    """
    วาดตาราง Heatmap ของ Confusion Matrix และบันทึกเป็นไฟล์ PNG
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")

    # กำหนดป้ายกำกับแกน X และแกน Y ตามชื่อคลาส
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("SVM Confusion Matrix")

    # ใส่ตัวเลขจำนวนภาพลงในแต่ละช่องของตาราง พร้อมสลับสีข้อความให้อ่านง่าย
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
