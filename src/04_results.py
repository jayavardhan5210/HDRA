import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models\ecg_cnn_improved_best.keras"
RESULT_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\results"

os.makedirs(RESULT_PATH, exist_ok=True)

print("=" * 60)
print("ECG MODEL RESULTS")
print("=" * 60)

print("\nLoading test data...")

X_test = np.load(
    os.path.join(
        DATA_PATH,
        "X_test.npy"
    )
)

Y_test = np.load(
    os.path.join(
        DATA_PATH,
        "Y_test.npy"
    )
)

CLASSES = np.load(
    os.path.join(
        DATA_PATH,
        "classes.npy"
    ),
    allow_pickle=True
)

print(
    "Test data:",
    X_test.shape
)

print(
    "Test labels:",
    Y_test.shape
)

print(
    "Classes:",
    CLASSES
)

print("\nLoading improved model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully."
)

print("\nGenerating predictions...")

Y_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

Y_pred = (
    Y_prob >= 0.5
).astype(int)

print(
    "\nPredictions generated."
)

print("\nCalculating test accuracy...")

test_accuracy = np.mean(
    Y_pred == Y_test
)

print(
    "Test Binary Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\n" + "=" * 60)
print("GENERATING CONFUSION MATRICES")
print("=" * 60)

for i, class_name in enumerate(CLASSES):

    cm = confusion_matrix(
        Y_test[:, i],
        Y_pred[:, i]
    )

    print(
        f"\n{class_name} confusion matrix:"
    )

    print(cm)

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        f"Confusion Matrix - {class_name}"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        [
            "Negative",
            "Positive"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "Negative",
            "Positive"
        ]
    )

    for row in range(2):

        for col in range(2):

            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    file_path = os.path.join(
        RESULT_PATH,
        f"confusion_matrix_{class_name}.png"
    )

    plt.savefig(
        file_path,
        dpi=300
    )

    plt.close()

print(
    "\nAll confusion matrices saved."
)

print("\n" + "=" * 60)
print("GENERATING ROC CURVE")
print("=" * 60)

plt.figure(
    figsize=(8, 6)
)

for i, class_name in enumerate(CLASSES):

    try:

        fpr, tpr, _ = roc_curve(
            Y_test[:, i],
            Y_prob[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        print(
            f"{class_name} AUC: "
            f"{roc_auc:.4f}"
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={roc_auc:.3f})"
            )
        )

    except ValueError:

        print(
            f"{class_name}: "
            "ROC could not be calculated"
        )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - ECG Classification"
)

plt.legend()

plt.tight_layout()

roc_path = os.path.join(
    RESULT_PATH,
    "roc_curves.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

print(
    "\nROC curve saved."
)

print("\n" + "=" * 60)
print("SAVING PREDICTIONS")
print("=" * 60)

np.save(
    os.path.join(
        RESULT_PATH,
        "Y_test_predictions.npy"
    ),
    Y_pred
)

np.save(
    os.path.join(
        RESULT_PATH,
        "Y_test_probabilities.npy"
    ),
    Y_prob
)

print(
    "Predictions saved."
)

print("\n" + "=" * 60)
print("RESULT FILES")
print("=" * 60)

for class_name in CLASSES:

    print(
        f"confusion_matrix_{class_name}.png"
    )

print(
    "roc_curves.png"
)

print(
    "Y_test_predictions.npy"
)

print(
    "Y_test_probabilities.npy"
)

print("\n" + "=" * 60)
print("RESULT GENERATION COMPLETED")
print("=" * 60)

print(
    f"\nFinal Test Binary Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nResults location:\n{RESULT_PATH}"
)