import json
import os
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 700
BATCH_SIZE = 32


def main():
    print("--" * 30)
    print("Neural Network: Rock vs Paper vs Scissors")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Compare NN Configurations
    print("\n" + "=" * 45)
    print("[Step 4] Comparing Neural Network Configurations")
    print("=" * 45)

    configs = ["Simple", "Medium", "Deep"]
    config_results = {}

    for cfg in configs:
        print(f"\n--- Testing Configuration: {cfg} ---")
        m, _ = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            config=cfg, epochs=20, batch_size=BATCH_SIZE
        )
        preds = predict_model(m, X_test)
        acc = evaluate_model(y_test, preds, classes)
        config_results[cfg] = acc

    # Step 5: Compare Epochs on Deep Configuration
    print("\n" + "=" * 45)
    print("[Step 5] Comparing Different Numbers of Epochs (Deep Model)")
    print("=" * 45)

    epochs_list = [10, 20, 30]
    epoch_results = {}
    best_model = None
    best_history = None
    best_accuracy = 0.0

    for ep in epochs_list:
        print(f"\n--- Training with Epochs: {ep} ---")
        m, hist = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            config="Deep", output_dir=None, epochs=ep, batch_size=BATCH_SIZE
        )
        preds = predict_model(m, X_test)
        acc = evaluate_model(y_test, preds, classes)
        epoch_results[ep] = acc

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = m
            best_history = hist

    # Step 6: Save the Best Model and Evaluation Plots
    print("\n" + "=" * 45)
    print(f"🏆 Best Model Accuracy: {best_accuracy * 100:.2f}%")
    print("=" * 45)

    best_model.save(os.path.join(OUTPUT_DIR, "nn_model.keras"))
    final_predictions = predict_model(best_model, X_test)

    evaluate_model(
        y_test, final_predictions, classes,
        save_path=f"{OUTPUT_DIR}/confusion_matrix.png"
    )
    plot_history(best_history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()