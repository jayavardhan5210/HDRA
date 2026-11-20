import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models"

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "ecg_cnn_best_improved.keras"
)

THRESHOLD_FILE = os.path.join(
    MODEL_PATH,
    "accuracy_thresholds.npy"
)

print("=" * 65)
print("OPTIMIZING THRESHOLDS FOR BINARY ACCURACY")
print("=" * 65)

print("\nLoading data...")

X_val = np.load(
    os.path.join(DATA_PATH, "X_val.npy")
)

Y_val = np.load(
    os.path.join(DATA_PATH, "Y_val.npy")
)

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

print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("Classes:", CLASSES)

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Model loaded successfully.")

print("\nGenerating validation predictions...")

val_pred = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

print("\nGenerating test predictions...")

test_pred = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

print("\nOptimizing class-specific thresholds...")

thresholds = []

for i, name in enumerate(CLASSES):

    best_threshold = 0.50
    best_accuracy = 0.0

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (
            val_pred[:, i] >= threshold
        ).astype(int)

        accuracy = np.mean(
            predictions == Y_val[:, i]
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    thresholds.append(best_threshold)

    print(
        f"{name}: "
        f"threshold={best_threshold:.2f}, "
        f"validation_accuracy={best_accuracy * 100:.2f}%"
    )

thresholds = np.array(thresholds)

np.save(
    THRESHOLD_FILE,
    thresholds
)

print("\nThresholds saved:")
print(THRESHOLD_FILE)

print("\n" + "=" * 65)
print("VALIDATION EVALUATION")
print("=" * 65)

val_binary = (
    val_pred >= thresholds
).astype(int)

val_accuracy = np.mean(
    val_binary == Y_val
)

print(
    f"Validation Binary Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_binary = (
    test_pred >= thresholds
).astype(int)

test_accuracy = np.mean(
    test_binary == Y_test
)

print(
    f"Test Binary Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        Y_test,
        test_binary,
        target_names=CLASSES,
        zero_division=0
    )
)

print("\nROC-AUC SCORES:")

auc_scores = []

for i, name in enumerate(CLASSES):

    auc = roc_auc_score(
        Y_test[:, i],
        test_pred[:, i]
    )

    auc_scores.append(auc)

    print(
        f"{name}: {auc:.4f}"
    )

mean_auc = np.mean(auc_scores)

print(
    f"\nMean ROC-AUC: {mean_auc:.4f}"
)

print("\n" + "=" * 65)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 65)