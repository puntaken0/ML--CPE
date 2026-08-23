import json
import os
import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = "data"          # โฟลเดอร์ที่เก็บ paper, rock, scissors
OUTPUT_DIR = "outputs"
IMG_SIZE = 100
TEST_SIZE = 0.2
MAX_PER_CLASS = 700         # จำกัดรูปเพื่อความเร็วในการประมวลผล
KERNELS = ["linear", "poly", "rbf"]

def main():
    print("--" * 30)
    print("SVM Image Recognition: Rock vs Paper vs Scissors")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Step 4 & 5: Train and Compare 3 SVM Kernels
    print("\n[Step 4] Training and Comparing SVM Kernels...")
    best_accuracy = 0.0
    best_kernel = None
    best_model = None
    best_scaler = None

    for k in KERNELS:
        print(f"\n--- Testing SVM Kernel: {k.upper()} ---")
        model, scaler = train_svm(X_train, y_train, kernel=k)
        predictions = predict_svm(model, scaler, X_test)
        
        acc = evaluate_model(
            y_test, predictions, classes,
            save_path=f"{OUTPUT_DIR}/confusion_matrix_{k}.png"
        )

        if acc > best_accuracy:
            best_accuracy = acc
            best_kernel = k
            best_model = model
            best_scaler = scaler

    # Step 6: Save the Best Model
    print("\n" + "=" * 45)
    print(f"Best SVM Kernel is: {best_kernel.upper()} (Accuracy: {best_accuracy * 100:.2f}%)")
    print("=" * 45)

    joblib.dump(best_model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(best_scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print(f"Saved best model ({best_kernel}) to {OUTPUT_DIR}/svm_model.pkl")

if __name__ == "__main__":
    main()