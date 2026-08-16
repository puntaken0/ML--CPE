import json
import os
import joblib
import numpy as np

from data_load import load_data
from preprocess import preprocess_mobile_data
from split_data import split_dataset
from knn_model import train_knn, predict_knn
from evaluate import evaluate_model

DATA_PATH = "Mobile.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2
K_VALUES = [3, 5, 7, 9, 11]  # ค่า k ที่ใช้ทดสอบเปรียบเทียบ

def main():
    print("--" * 30)
    print("KNN Classification: Mobile Price Range Prediction")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    df, labels, classes = load_data(DATA_PATH)

    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing features...")
    X = preprocess_mobile_data(df)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    # Step 4 & 5: Train and Compare KNN with multiple k values
    print("\n[Step 4] Training and Comparing KNN with different k values...")
    best_accuracy = 0.0
    best_k = None
    best_model = None
    best_scaler = None

    for k in K_VALUES:
        print(f"\n--- Testing KNN with k = {k} ---")
        model, scaler = train_knn(X_train, y_train, n_neighbors=k)
        predictions = predict_knn(model, scaler, X_test)
        
        acc = evaluate_model(
            y_test, predictions, classes,
            save_path=f"{OUTPUT_DIR}/confusion_matrix_k{k}.png"
        )

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            best_model = model
            best_scaler = scaler

    # Step 6: Save the Best Model
    print("\n" + "=" * 40)
    print(f"🏆 Best k value is: k = {best_k} (Accuracy: {best_accuracy * 100:.2f}%)")
    print("=" * 40)

    joblib.dump(best_model, f"{OUTPUT_DIR}/knn_model.pkl")
    joblib.dump(best_scaler, f"{OUTPUT_DIR}/scaler.pkl")
    print(f"Saved best model (k={best_k}) to {OUTPUT_DIR}/knn_model.pkl")

if __name__ == "__main__":
    main()