import json
import os
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore


def build_dcnn_model(input_shape, num_classes, config="DCNN-4Blocks"):
    """
    สร้างโมเดล Deep CNN ตามระดับความลึก:
    - DCNN-3Blocks: 3 Conv Blocks (Filters: 32 -> 64 -> 128)
    - DCNN-4Blocks: 4 Conv Blocks (Filters: 32 -> 64 -> 128 -> 256)
    - DCNN-DeepDouble: 6 Conv Layers (ซ้อนคู่ 32x2 -> 64x2 -> 128x2 + GlobalAveragePooling)
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))

    if config == "DCNN-3Blocks":
        for filters in [32, 64, 128]:
            model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.4))

    elif config == "DCNN-4Blocks":
        for filters in [32, 64, 128, 256]:
            model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))

    else:  # "DCNN-DeepDouble"
        for filters in [32, 64, 128]:
            model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
            model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(0.3))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.4))

    # Output Layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_dcnn(X_train, y_train, X_val, y_val, num_classes,
               config="DCNN-4Blocks", output_dir=None, epochs=20, batch_size=32):
    model = build_dcnn_model(X_train.shape[1:], num_classes, config=config)

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
        model.save(os.path.join(output_dir, "dcnn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)

    return model, history


def predict_dcnn(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    return probabilities.argmax(axis=1)