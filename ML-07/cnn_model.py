import json
import os
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore


def build_cnn_model(input_shape, num_classes, config="Deep"):
    """
    สร้างโมเดล CNN ตามระดับความลึกของสถาปัตยกรรม:
    - Simple: 1 Conv Layer Block
    - Medium: 2 Conv Layer Blocks
    - Deep: 3 Conv Layer Blocks พร้อม BatchNormalization & Dropout
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    if config == "Simple":
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.2))

    elif config == "Medium":
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))

    else:  # "Deep" (3 Conv Blocks)
        # Block 1
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 2
        model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # Block 3
        model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(0.3))

    # Output Layer สำหรับจำแนก 3 คลาส
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(X_train, y_train, X_val, y_val, num_classes,
              config="Deep", output_dir=None, epochs=20, batch_size=32):
    model = build_cnn_model(X_train.shape[1:], num_classes, config=config)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "cnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

    return model, history


def predict_cnn(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int)
    return probabilities.argmax(axis=1)
