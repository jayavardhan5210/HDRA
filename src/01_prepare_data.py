import os
import ast
import numpy as np
import pandas as pd
import wfdb

BASE_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\ptb-xl-a-dataset"

DATABASE_FILE = os.path.join(
    BASE_PATH,
    "ptbxl_database.csv"
)

SCP_FILE = os.path.join(
    BASE_PATH,
    "scp_statements.csv"
)

OUTPUT_PATH = r"C:\Users\balum\OneDrive\Desktop\hdra\dataset\processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("PTB-XL ECG DATA PREPARATION")
print("=" * 60)

if not os.path.exists(DATABASE_FILE):
    raise FileNotFoundError(
        f"ptbxl_database.csv not found: {DATABASE_FILE}"
    )

if not os.path.exists(SCP_FILE):
    raise FileNotFoundError(
        f"scp_statements.csv not found: {SCP_FILE}"
    )

print("\nLoading PTB-XL database...")

df = pd.read_csv(
    DATABASE_FILE,
    index_col="ecg_id"
)

print("Total ECG records in CSV:", len(df))

df["scp_codes"] = df["scp_codes"].apply(
    ast.literal_eval
)

print("\nLoading SCP statements...")

scp = pd.read_csv(
    SCP_FILE,
    index_col=0
)

scp = scp[
    scp["diagnostic"] == 1
]

CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "CD"
]

print("\nClasses:")

for i, class_name in enumerate(CLASSES):
    print(i, "->", class_name)


def get_labels(code_dict):
    labels = []

    for code in code_dict.keys():

        if code in scp.index:

            diagnostic_class = scp.loc[
                code,
                "diagnostic_class"
            ]

            if diagnostic_class in CLASSES:
                labels.append(diagnostic_class)

    return list(set(labels))


df["diagnostic_classes"] = df[
    "scp_codes"
].apply(
    get_labels
)

df = df[
    df["diagnostic_classes"].apply(len) > 0
].copy()

print(
    "\nRecords with diagnostic labels:",
    len(df)
)


def encode_labels(labels):

    result = np.zeros(
        len(CLASSES),
        dtype=np.float32
    )

    for label in labels:

        if label in CLASSES:

            index = CLASSES.index(label)

            result[index] = 1.0

    return result


df["label_vector"] = df[
    "diagnostic_classes"
].apply(
    encode_labels
)


def load_ecg(record_path):

    signal, metadata = wfdb.rdsamp(
        record_path
    )

    signal = signal.astype(
        np.float32
    )

    signal = np.nan_to_num(
        signal
    )

    mean = np.mean(
        signal,
        axis=0,
        keepdims=True
    )

    std = np.std(
        signal,
        axis=0,
        keepdims=True
    )

    std[std == 0] = 1

    signal = (
        signal - mean
    ) / std

    return signal


X = []
Y = []
folds = []

missing_records = []
failed_records = []

print("\n" + "=" * 60)
print("Loading ECG signals...")
print("=" * 60)

total_records = len(df)

for counter, (ecg_id, row) in enumerate(
    df.iterrows()
):

    record_path = os.path.join(
        BASE_PATH,
        row["filename_lr"]
    )

    header_file = record_path + ".hea"
    data_file = record_path + ".dat"

    if (
        not os.path.exists(header_file)
        or not os.path.exists(data_file)
    ):

        missing_records.append(
            ecg_id
        )

        continue

    try:

        signal = load_ecg(
            record_path
        )

        X.append(signal)

        Y.append(
            row["label_vector"]
        )

        folds.append(
            row["strat_fold"]
        )

    except Exception:

        failed_records.append(
            ecg_id
        )

        continue

    if (counter + 1) % 500 == 0:

        print(
            f"Processed {counter + 1}/{total_records}"
        )


print("\nConverting data to NumPy arrays...")

X = np.array(
    X,
    dtype=np.float32
)

Y = np.array(
    Y,
    dtype=np.float32
)

folds = np.array(
    folds
)

print("\n" + "=" * 60)
print("DATA LOADING COMPLETED")
print("=" * 60)

print(
    "Successfully loaded ECG records:",
    len(X)
)

print(
    "Missing ECG records:",
    len(missing_records)
)

print(
    "Failed ECG records:",
    len(failed_records)
)

print("\nFinal data shape:")

print(
    "X:",
    X.shape
)

print(
    "Y:",
    Y.shape
)

print("\nCreating dataset splits...")

train_mask = folds < 9
val_mask = folds == 9
test_mask = folds == 10

X_train = X[train_mask]
Y_train = Y[train_mask]

X_val = X[val_mask]
Y_val = Y[val_mask]

X_test = X[test_mask]
Y_test = Y[test_mask]

print("\nDataset split:")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)

print("\nSaving processed data...")

np.save(
    os.path.join(
        OUTPUT_PATH,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "Y_train.npy"
    ),
    Y_train
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "Y_val.npy"
    ),
    Y_val
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "Y_test.npy"
    ),
    Y_test
)

np.save(
    os.path.join(
        OUTPUT_PATH,
        "classes.npy"
    ),
    np.array(CLASSES)
)

if len(missing_records) > 0:

    missing_file = os.path.join(
        OUTPUT_PATH,
        "missing_records.txt"
    )

    with open(
        missing_file,
        "w"
    ) as f:

        for ecg_id in missing_records:
            f.write(
                str(ecg_id) + "\n"
            )

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(
    "\nProcessed files saved in:"
)

print(
    OUTPUT_PATH
)

print("\nCreated files:")

print("X_train.npy")
print("Y_train.npy")
print("X_val.npy")
print("Y_val.npy")
print("X_test.npy")
print("Y_test.npy")
print("classes.npy")

if len(missing_records) > 0:

    print(
        "\nMissing records skipped:",
        len(missing_records)
    )

print("\nYou can now proceed to model training.")