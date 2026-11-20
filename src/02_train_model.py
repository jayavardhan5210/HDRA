import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models"
RESULT_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\results"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

print("=" * 70)
print("IMPROVED ECG CNN TRAINING")
print("=" * 70)

print("\nLoading dataset...")

X_train = np.load(
    os.path.join(DATA_PATH, "X_train.npy")
)

Y_train = np.load(
    os.path.join(DATA_PATH, "Y_train.npy")
)

X_val = np.load(
    os.path.join(DATA_PATH, "X_val.npy")
)

Y_val = np.load(
    os.path.join(DATA_PATH, "Y_val.npy")
)

CLASSES = np.load(
    os.path.join(DATA_PATH, "classes.npy"),
    allow_pickle=True
)

print("Training data :", X_train.shape)
print("Training labels:", Y_train.shape)
print("Validation data:", X_val.shape)
print("Validation labels:", Y_val.shape)
print("Classes:", CLASSES)

print("\nChecking dataset...")

print("NaN in X_train:", np.isnan(X_train).sum())
print("NaN in X_val:", np.isnan(X_val).sum())

X_train = np.nan_to_num(X_train).astype(np.float32)
X_val = np.nan_to_num(X_val).astype(np.float32)

Y_train = Y_train.astype(np.float32)
Y_val = Y_val.astype(np.float32)

print("\nCalculating class statistics...")

positive_counts = np.sum(Y_train, axis=0)
total_samples = len(Y_train)

print("\nClass distribution:")

for i, name in enumerate(CLASSES):
    percentage = (
        positive_counts[i] / total_samples
    ) * 100

    print(
        f"{name}: "
        f"{int(positive_counts[i])} "
        f"({percentage:.2f}%)"
    )


print("\nCalculating positive class weights...")

negative_counts = total_samples - positive_counts

pos_weights = (
    negative_counts /
    (positive_counts + 1e-7)
)

pos_weights = np.sqrt(pos_weights)

pos_weights = (
    pos_weights /
    np.mean(pos_weights)
)

pos_weights = np.clip(
    pos_weights,
    0.75,
    2.5
)

print("\nPositive class weights:")

for i, name in enumerate(CLASSES):
    print(
        f"{name}: "
        f"{pos_weights[i]:.3f}"
    )


POS_WEIGHTS = tf.constant(
    pos_weights,
    dtype=tf.float32
)


def weighted_focal_loss(
    y_true,
    y_pred
):
    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7
    )

    alpha = POS_WEIGHTS

    positive_loss = (
        -alpha *
        y_true *
        tf.pow(
            1.0 - y_pred,
            2.0
        ) *
        tf.math.log(y_pred)
    )

    negative_loss = (
        -(1.0 - y_true) *
        tf.pow(
            y_pred,
            2.0
        ) *
        tf.math.log(
            1.0 - y_pred
        )
    )

    loss = (
        positive_loss +
        negative_loss
    )

    return tf.reduce_mean(loss)


def augment_ecg(
    signal,
    label
):
    noise = tf.random.normal(
        tf.shape(signal),
        mean=0.0,
        stddev=0.01
    )

    signal = signal + noise

    scale = tf.random.uniform(
        [],
        minval=0.95,
        maxval=1.05
    )

    signal = signal * scale

    return signal, label


print("\nCreating TensorFlow datasets...")

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    reshuffle_each_iteration=True
)

train_dataset = train_dataset.map(
    augment_ecg,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.batch(
    32
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, Y_val)
)

val_dataset = val_dataset.batch(
    32
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("\nBuilding CNN model...")

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Conv1D(
        64,
        kernel_size=7,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(
        0.15
    ),

    Conv1D(
        128,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(
        0.20
    ),

    Conv1D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(
        0.25
    ),

    Conv1D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv1D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    GlobalAveragePooling1D(),

    Dense(
        256,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(
        0.35
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.25
    ),

    Dense(
        len(CLASSES),
        activation="sigmoid"
    )
])


optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0003
)


model.compile(
    optimizer=optimizer,
    loss=weighted_focal_loss,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        ),
        tf.keras.metrics.AUC(
            name="auc",
            multi_label=True,
            num_labels=len(CLASSES)
        )
    ]
)


print("\nModel summary:")

model.summary()


callbacks = [

    ModelCheckpoint(
        os.path.join(
            MODEL_PATH,
            "ecg_cnn_best_improved.keras"
        ),
        monitor="val_binary_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)


print("\nSaving final model...")

final_model_path = os.path.join(
    MODEL_PATH,
    "ecg_cnn_improved_final.keras"
)

model.save(
    final_model_path
)

print(
    "Final model saved:",
    final_model_path
)


print("\nGenerating accuracy graph...")

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    history.history["binary_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_binary_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Accuracy")

plt.title(
    "ECG CNN Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

accuracy_path = os.path.join(
    RESULT_PATH,
    "improved_accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


print("\nGenerating loss graph...")

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ECG CNN Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

loss_path = os.path.join(
    RESULT_PATH,
    "improved_loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


print("\nGenerating AUC graph...")

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    history.history["auc"],
    label="Training AUC"
)

plt.plot(
    history.history["val_auc"],
    label="Validation AUC"
)

plt.xlabel("Epoch")
plt.ylabel("AUC")

plt.title(
    "ECG CNN Training and Validation AUC"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

auc_path = os.path.join(
    RESULT_PATH,
    "improved_auc_curve.png"
)

plt.savefig(
    auc_path,
    dpi=300
)

plt.close()


best_train_accuracy = max(
    history.history["binary_accuracy"]
)

best_val_accuracy = max(
    history.history["val_binary_accuracy"]
)

best_train_auc = max(
    history.history["auc"]
)

best_val_auc = max(
    history.history["val_auc"]
)

best_train_loss = min(
    history.history["loss"]
)

best_val_loss = min(
    history.history["val_loss"]
)


print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Training Accuracy : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best Training AUC      : "
    f"{best_train_auc:.4f}"
)

print(
    f"Best Validation AUC    : "
    f"{best_val_auc:.4f}"
)

print(
    f"Best Training Loss     : "
    f"{best_train_loss:.4f}"
)

print(
    f"Best Validation Loss   : "
    f"{best_val_loss:.4f}"
)

print("\nBest model:")
print(
    os.path.join(
        MODEL_PATH,
        "ecg_cnn_best_improved.keras"
    )
)

print("\nFinal model:")
print(final_model_path)

print("\nGraphs:")

print(accuracy_path)
print(loss_path)
print(auc_path)

print("\nNext step:")
print("Run the test evaluation using the new best model.")

print("=" * 70)