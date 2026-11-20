import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models"

X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_PATH, "Y_val.npy"))

CLASSES = np.load(
    os.path.join(DATA_PATH, "classes.npy"),
    allow_pickle=True
)

model = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, "ecg_cnn_improved_best.keras"),
    compile=False
)

print("=" * 60)
print("OPTIMIZING CLASS-SPECIFIC THRESHOLDS")
print("=" * 60)

print("\nGenerating validation predictions...")

Y_prob = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

thresholds = []

for i, class_name in enumerate(CLASSES):

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in np.arange(0.20, 0.81, 0.01):

        pred = (
            Y_prob[:, i] >= threshold
        ).astype(int)

        score = f1_score(
            Y_val[:, i],
            pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    thresholds.append(best_threshold)

    print(
        f"{class_name}: "
        f"threshold={best_threshold:.2f}, "
        f"F1={best_f1:.4f}"
    )

thresholds = np.array(thresholds)

threshold_path = os.path.join(
    MODEL_PATH,
    "class_thresholds.npy"
)

np.save(
    threshold_path,
    thresholds
)

print("\n" + "=" * 60)
print("OPTIMAL THRESHOLDS")
print("=" * 60)

for name, threshold in zip(CLASSES, thresholds):
    print(
        f"{name}: {threshold:.2f}"
    )

print("\nSaved at:")
print(threshold_path)