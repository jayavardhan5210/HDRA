import os
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\models\ecg_cnn_improved_best.keras"

model = load_model(
    MODEL_PATH,
    compile=False
)

classes = np.array([
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "CD"
])

print("=" * 60)
print("ECG DISEASE PREDICTION")
print("=" * 60)

print("\nModel loaded successfully.")

file_path = input(
    "\nEnter ECG .npy file path: "
).strip()

if not os.path.exists(file_path):
    print("\nFile not found.")
    print("Please enter a valid .npy ECG file path.")
    exit()

try:

    ecg = np.load(file_path)

    print(
        "\nOriginal ECG shape:",
        ecg.shape
    )

    if ecg.ndim == 2:
        ecg = np.expand_dims(
            ecg,
            axis=0
        )

    if ecg.shape[1:] != (1000, 12):

        print(
            "\nInvalid ECG shape."
        )

        print(
            "Expected: (1000, 12)"
        )

        print(
            "Received:",
            ecg.shape[1:]
        )

        exit()

    print(
        "\nRunning prediction..."
    )

    probabilities = model.predict(
        ecg,
        verbose=0
    )[0]

    print(
        "\n" + "=" * 60
    )

    print(
        "PREDICTION RESULTS"
    )

    print(
        "=" * 60
    )

    for i in range(len(classes)):

        print(
            f"{classes[i]:6s} : "
            f"{probabilities[i] * 100:.2f}%"
        )

    best_index = np.argmax(
        probabilities
    )

    best_class = classes[
        best_index
    ]

    print(
        "\n" + "=" * 60
    )

    print(
        "MOST LIKELY CLASS:",
        best_class
    )

    print(
        "CONFIDENCE:",
        f"{probabilities[best_index] * 100:.2f}%"
    )

    print(
        "=" * 60
    )

except Exception as e:

    print(
        "\nError during prediction:"
    )

    print(e)