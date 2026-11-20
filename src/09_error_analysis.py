import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
    roc_auc_score
)

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models"
RESULT_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\results"

os.makedirs(RESULT_PATH, exist_ok=True)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "ecg_cnn_best_improved.keras"
)

THRESHOLD_FILE = os.path.join(
    MODEL_PATH,
    "accuracy_thresholds.npy"
)

print("=" * 65)
print("ECG MODEL ERROR ANALYSIS")
print("=" * 65)

print("\nLoading data...")

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
    THRESHOLD_FILE
)

print("Test data:", X_test.shape)
print("Classes:", CLASSES)
print("Thresholds:", thresholds)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

print("Model loaded successfully.")

print("\nGenerating predictions...")

pred_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

pred = (
    pred_prob >= thresholds
).astype(int)

print("\nCalculating confusion matrices...")

cm = multilabel_confusion_matrix(
    Y_test,
    pred
)

print("\n" + "=" * 65)
print("CLASS-WISE RESULTS")
print("=" * 65)

class_accuracy = []
class_precision = []
class_recall = []
class_f1 = []
class_auc = []

for i, name in enumerate(CLASSES):

    tn, fp, fn, tp = cm[i].ravel()

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
    )

    precision = (
        tp /
        (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    auc = roc_auc_score(
        Y_test[:, i],
        pred_prob[:, i]
    )

    class_accuracy.append(accuracy)
    class_precision.append(precision)
    class_recall.append(recall)
    class_f1.append(f1)
    class_auc.append(auc)

    print("\n" + str(name))
    print("-" * 30)
    print("True Negative :", tn)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Positive :", tp)
    print(f"Accuracy      : {accuracy * 100:.2f}%")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")

print("\nGenerating confusion matrix graph...")

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()

for i, name in enumerate(CLASSES):

    tn, fp, fn, tp = cm[i].ravel()

    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    axes[i].imshow(matrix)

    axes[i].set_title(str(name))

    axes[i].set_xlabel(
        "Predicted"
    )

    axes[i].set_ylabel(
        "Actual"
    )

    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])

    axes[i].set_xticklabels(
        ["Negative", "Positive"]
    )

    axes[i].set_yticklabels(
        ["Negative", "Positive"]
    )

    for row in range(2):
        for col in range(2):
            axes[i].text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center"
            )

axes[-1].axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "multilabel_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion matrix saved.")

print("\nGenerating performance graph...")

x = np.arange(len(CLASSES))
width = 0.2

plt.figure(figsize=(11, 6))

plt.bar(
    x - 1.5 * width,
    class_accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    class_precision,
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    class_recall,
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    class_f1,
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    CLASSES
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "ECG Classes"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Class-wise Performance of ECG CNN Model"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "classwise_performance.png"
    ),
    dpi=300
)

plt.close()

print("Performance graph saved.")

print("\nGenerating ROC-AUC graph...")

plt.figure(figsize=(9, 6))

plt.bar(
    CLASSES,
    class_auc
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "ECG Classes"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "Class-wise ROC-AUC Performance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "classwise_roc_auc.png"
    ),
    dpi=300
)

plt.close()

print("ROC-AUC graph saved.")

print("\nGenerating error analysis graph...")

false_positive = []
false_negative = []

for i in range(len(CLASSES)):

    tn, fp, fn, tp = cm[i].ravel()

    false_positive.append(fp)
    false_negative.append(fn)

plt.figure(figsize=(10, 6))

x = np.arange(len(CLASSES))

plt.bar(
    x - 0.2,
    false_positive,
    0.4,
    label="False Positives"
)

plt.bar(
    x + 0.2,
    false_negative,
    0.4,
    label="False Negatives"
)

plt.xticks(
    x,
    CLASSES
)

plt.xlabel(
    "ECG Classes"
)

plt.ylabel(
    "Number of Errors"
)

plt.title(
    "Class-wise Prediction Errors"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_PATH,
        "classwise_errors.png"
    ),
    dpi=300
)

plt.close()

print("Error graph saved.")

print("\n" + "=" * 65)
print("ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated files:")

print(
    os.path.join(
        RESULT_PATH,
        "multilabel_confusion_matrix.png"
    )
)

print(
    os.path.join(
        RESULT_PATH,
        "classwise_performance.png"
    )
)

print(
    os.path.join(
        RESULT_PATH,
        "classwise_roc_auc.png"
    )
)

print(
    os.path.join(
        RESULT_PATH,
        "classwise_errors.png"
    )
)