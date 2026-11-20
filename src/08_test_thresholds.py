import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    roc_auc_score
)

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models"

print("=" * 60)
print("TEST EVALUATION WITH CLASS-SPECIFIC THRESHOLDS")
print("=" * 60)

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

Y_test = np.load(
    os.path.join(DATA_PATH, "Y_test.npy")
)

CLASSES = np.load(
    os.path.join(DATA_PATH, "classes.npy"),
    allow_pickle=True
)

thresholds = np.load(
    os.path.join(
        MODEL_PATH,
        "class_thresholds.npy"
    )
)

model = tf.keras.models.load_model(
    os.path.join(
        MODEL_PATH,
        "ecg_cnn_improved_best.keras"
    ),
    compile=False
)

print("\nTest data:", X_test.shape)
print("Classes:", CLASSES)

print("\nClass-specific thresholds:")

for name, threshold in zip(CLASSES, thresholds):
    print(
        f"{name}: {threshold:.2f}"
    )

print("\nGenerating test predictions...")

Y_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

Y_pred = (
    Y_prob >= thresholds
).astype(int)

binary_accuracy = np.mean(
    Y_pred == Y_test
)

subset_accuracy = np.mean(
    np.all(
        Y_pred == Y_test,
        axis=1
    )
)

hamming = hamming_loss(
    Y_test,
    Y_pred
)

weighted_f1 = f1_score(
    Y_test,
    Y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0
)

micro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Binary Accuracy: "
    f"{binary_accuracy * 100:.2f}%"
)

print(
    f"Subset Accuracy: "
    f"{subset_accuracy * 100:.2f}%"
)

print(
    f"Hamming Loss: "
    f"{hamming:.4f}"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1: "
    f"{weighted_f1:.4f}"
)

print(
    f"Micro F1: "
    f"{micro_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        Y_test,
        Y_pred,
        target_names=CLASSES,
        zero_division=0
    )
)

print("\nROC-AUC SCORES:")

roc_scores = []

for i, name in enumerate(CLASSES):

    try:
        score = roc_auc_score(
            Y_test[:, i],
            Y_prob[:, i]
        )

        roc_scores.append(score)

        print(
            f"{name}: {score:.4f}"
        )

    except ValueError:
        print(
            f"{name}: unavailable"
        )

if roc_scores:

    mean_auc = np.mean(
        roc_scores
    )

    print(
        f"\nMean ROC-AUC: "
        f"{mean_auc:.4f}"
    )

print("\n" + "=" * 60)
print("THRESHOLD-BASED EVALUATION COMPLETED")
print("=" * 60)