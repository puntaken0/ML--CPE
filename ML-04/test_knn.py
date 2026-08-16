import json
import joblib
import numpy as np

OUTPUT_DIR = "outputs"
N_SAMPLES = 4

def test_knn(n_samples=N_SAMPLES):
    model = joblib.load(f"{OUTPUT_DIR}/knn_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # สุ่มตัวอย่างข้อมูล
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # ทำนายผล
    predictions = model.predict(scaler.transform(X_sample))

    print("\n=== KNN Prediction Test (Mobile Price Range) ===")
    for i in range(n_samples):
        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        status = "CORRECT" if predictions[i] == y_sample[i] else "WRONG"
        print(f"[{i + 1}] Predicted: {pred:<15} | Actual: {true:<15} -> {status}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nSummary: {correct_total}/{n_samples} correct ({correct_total/n_samples * 100:.1f}%)")

if __name__ == "__main__":
    test_knn()