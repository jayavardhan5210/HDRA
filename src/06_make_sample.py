import os
import numpy as np

DATA_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"

X_test = np.load(
    os.path.join(DATA_PATH, "X_test.npy")
)

sample = X_test[0]

output_path = os.path.join(
    DATA_PATH,
    "sample_ecg.npy"
)

np.save(
    output_path,
    sample
)

print("Sample ECG created successfully!")
print("Shape:", sample.shape)
print("File:")
print(output_path)