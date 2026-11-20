import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"
MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models\ecg_cnn_best_improved.keras"

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

print("=" * 60)
print("ECG MODEL EVALUATION")
print("=" * 60)

print("\nLoading improved model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print("\nGenerating validation predictions...")

Y_val_prob = model.predict(
    X_val,
    batch_size=32,
    verbose=1
)

thresholds = np.arange(
    0.30,
    0.71,
    0.01
)

best_threshold = 0.50
best_accuracy = 0.0

for threshold in thresholds:

    Y_val_pred = (
        Y_val_prob >= threshold
    ).astype(int)

    accuracy = np.mean(
        Y_val_pred == Y_val
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold

print("\nBest threshold:", best_threshold)

print(
    "Validation Binary Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)

print("\nGenerating test predictions...")

Y_test_prob = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

Y_pred = (
    Y_test_prob >= best_threshold
).astype(int)

test_accuracy = np.mean(
    Y_pred == Y_test
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    "Test Binary Accuracy:",
    f"{test_accuracy * 100:.2f}%"
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

auc_scores = []

for i, class_name in enumerate(CLASSES):

    try:

        score = roc_auc_score(
            Y_test[:, i],
            Y_test_prob[:, i]
        )

        auc_scores.append(
            score
        )

        print(
            f"{class_name}: {score:.4f}"
        )

    except ValueError:

        print(
            f"{class_name}: Cannot calculate"
        )

if len(auc_scores) > 0:

    print(
        "\nMean ROC-AUC:",
        f"{np.mean(auc_scores):.4f}"
    )

np.save(
    os.path.join(
        DATA_PATH,
        "best_threshold.npy"
    ),
    np.array(
        [best_threshold]
    )
)

print("\nThreshold saved successfully.")
print("\nEvaluation completed successfully!")