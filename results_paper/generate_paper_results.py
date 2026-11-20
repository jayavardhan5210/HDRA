"""
============================================================================
 HEART DISEASE RISK ASSESSMENT AGENT
 Research-Paper Results & Figures Generator   (READ-ONLY / ANALYSIS ONLY)
----------------------------------------------------------------------------
This script NEVER writes into the project source, dataset, models or
results directories.  It only READS the existing artifacts and writes the
paper-ready output into  results_paper/ .

Data it reads (all pre-existing, untouched):
  dataset/processed/X_test.npy , Y_test.npy
  dataset/processed/X_val.npy  , Y_val.npy
  dataset/processed/Y_train.npy
  dataset/processed/classes.npy
  dataset/processed/best_thresholds.npy   (per-class validation sweep)
  dataset/processed/best_threshold.npy    (single global threshold)
  results/Y_test_probabilities.npy        (saved sigmoid probabilities)
  models/ecg_cnn_improved_best.keras      (used only if TensorFlow is present)
  results/improved_accuracy_curve.png , improved_loss_curve.png

Output (written ONLY into results_paper/):
  figures/fig1_training_validation_accuracy.png
  figures/fig2_training_validation_loss.png
  figures/fig3_confusion_matrix_ovr.png
  figures/fig4_classwise_precision_recall_f1.png
  figures/fig5_class_distribution.png
  figures/fig6_roc_curves.png
  figures/fig7_precision_recall_curves.png
  figures/fig8_overall_performance.png
  tables/table1_dataset_class_distribution.{csv,md}
  tables/table2_classification_report.{csv,md}
  tables/table3_overall_evaluation_metrics.{csv,md}
  tables/metrics_all.json
  SECTION4_RESULTS_AND_DISCUSSION_autofilled.md

HOW TO RUN (Windows, uses the project's own virtual environment):
  double-click  results_paper\\run_evaluation.bat
      -or-    from the project root:
  ecg_env\\Scripts\\python.exe results_paper\\generate_paper_results.py

Nothing here retrains, re-splits, re-labels or modifies the model.
============================================================================
"""

import os
import sys
import json
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Paths — script lives in results_paper/, project root is its parent
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)

DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = HERE
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DEFAULT_CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
NOT_ENOUGH = "Insufficient data available to calculate this metric."

COLORS = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def log(msg):
    print(msg)
    sys.stdout.flush()


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    log("  saved -> figures/%s" % name)


def pct(x):
    return "%.2f%%" % (100.0 * x)


# ===========================================================================
# 1. LOAD THE ACTUAL PROJECT DATA
# ===========================================================================
log("=" * 74)
log(" HEART DISEASE RISK ASSESSMENT AGENT - PAPER RESULTS GENERATOR")
log(" (analysis only: the project is never modified)")
log("=" * 74)

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_DIR, "Y_val.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))

Y_test = np.asarray(Y_test).astype(int)
Y_val = np.asarray(Y_val).astype(int)
Y_train = np.asarray(Y_train).astype(int)

# real class names as saved by src/01_prepare_data.py
classes_path = os.path.join(DATA_DIR, "classes.npy")
if os.path.exists(classes_path):
    CLASSES = [str(c) for c in np.load(classes_path, allow_pickle=True).tolist()]
else:
    CLASSES = list(DEFAULT_CLASSES)

N_CLASSES = len(CLASSES)

log("")
log("Split sizes:")
log("  train : %d records, %s" % (len(Y_train), Y_train.shape))
log("  val   : %d records, %s" % (len(Y_val), Y_val.shape))
log("  test  : %d records, %s" % (len(Y_test), Y_test.shape))
log("  classes: %s" % ", ".join(CLASSES))

# --- decision thresholds (exactly those selected during evaluation) --------
thresholds = np.full(N_CLASSES, 0.50, dtype=float)
thr_source = "default 0.50"
path_per_class = os.path.join(DATA_DIR, "best_thresholds.npy")
path_global = os.path.join(DATA_DIR, "best_threshold.npy")
if os.path.exists(path_per_class):
    thresholds = np.asarray(np.load(path_per_class), dtype=float).ravel()
    if len(thresholds) != N_CLASSES:
        thresholds = np.full(N_CLASSES, float(thresholds[0]))
    thr_source = "best_thresholds.npy (per-class validation sweep 0.30-0.70)"
elif os.path.exists(path_global):
    g = float(np.asarray(np.load(path_global)).ravel()[0])
    thresholds = np.full(N_CLASSES, g)
    thr_source = "best_threshold.npy (single global validation threshold)"

log("")
log("Decision thresholds [%s]:" % thr_source)
for i, c in enumerate(CLASSES):
    log("  %-5s : %.2f" % (c, thresholds[i]))

# ===========================================================================
# 2. TEST PROBABILITIES (reuse saved; never regenerate blindly)
# ===========================================================================
Y_prob = None
Y_val_prob = None
prob_source = ""

saved_prob_path = os.path.join(RESULT_DIR, "Y_test_probabilities.npy")
if os.path.exists(saved_prob_path):
    saved = np.load(saved_prob_path)
    saved = np.asarray(saved, dtype=float)
    if saved.shape == Y_test.shape:
        Y_prob = saved
        prob_source = "results/Y_test_probabilities.npy (saved by src/04_results.py)"

# Optionally recompute with the real model *if* TensorFlow is available.
# This does NOT change anything on disk - it is a pure read/inference step.
recomputed = False
try:
    from tensorflow.keras.models import load_model

    model_file = os.path.join(MODEL_DIR, "ecg_cnn_improved_best.keras")
    if os.path.exists(model_file):
        log("")
        log("TensorFlow detected - re-running inference with the trained model")
        log("(read-only, nothing is re-trained or overwritten):")
        log("  %s" % model_file)
        model = load_model(model_file, compile=False)
        Y_prob_new = np.asarray(model.predict(X_test, batch_size=32, verbose=0), dtype=float)
        if Y_prob is None:
            Y_prob = Y_prob_new
            prob_source = "re-computed from models/ecg_cnn_improved_best.keras"
        else:
            diff = float(np.max(np.abs(Y_prob_new - Y_prob)))
            log("  cross-check vs saved probabilities: max|diff| = %.3e" % diff)
        Y_val_prob = np.asarray(model.predict(X_val, batch_size=32, verbose=0), dtype=float)
        recomputed = True
except Exception as exc:  # TensorFlow absent -> saved probabilities are enough
    log("")
    log("TensorFlow not usable here (%s)" % str(exc).splitlines()[0])
    log("-> using the probabilities already saved in results/")

if Y_prob is None:
    raise SystemExit(
        "FATAL: no probabilities available. Neither results/Y_test_probabilities.npy "
        "nor a loadable TensorFlow model was found, so no real evaluation is possible."
    )

if Y_prob.shape != Y_test.shape:
    raise SystemExit(
        "FATAL: probability shape %s != Y_test shape %s" % (Y_prob.shape, Y_test.shape)
    )

log("")
log("Probabilities source: %s" % prob_source)

# ===========================================================================
# 3. MULTI-LABEL DECISION + AGGREGATE METRICS
# ===========================================================================
Y_pred = (Y_prob >= thresholds).astype(int)

hamming = float(np.mean(Y_pred != Y_test))
element_acc = 1.0 - hamming                      # this is the project's "test accuracy"
subset_acc = float(np.mean(np.all(Y_pred == Y_test, axis=1)))

micro_p = float(precision_score(Y_test, Y_pred, average="micro", zero_division=0))
micro_r = float(recall_score(Y_test, Y_pred, average="micro", zero_division=0))
micro_f = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
macro_p = float(precision_score(Y_test, Y_pred, average="macro", zero_division=0))
macro_r = float(recall_score(Y_test, Y_pred, average="macro", zero_division=0))
macro_f = float(f1_score(Y_test, Y_pred, average="macro", zero_division=0))
wei_p = float(precision_score(Y_test, Y_pred, average="weighted", zero_division=0))
wei_r = float(recall_score(Y_test, Y_pred, average="weighted", zero_division=0))
wei_f = float(f1_score(Y_test, Y_pred, average="weighted", zero_division=0))

# ===========================================================================
# 4. PER-CLASS METRICS, SENSITIVITY, SPECIFICITY
# ===========================================================================
per_class = {}
for i, c in enumerate(CLASSES):
    yt = Y_test[:, i]
    yp = Y_pred[:, i]
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    per_class[c] = {
        "precision": prec,
        "recall": rec,
        "sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "support": int(np.sum(yt == 1)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(thresholds[i]),
    }

macro_spec = float(np.mean([per_class[c]["specificity"] for c in CLASSES]))
tot_tn = sum(per_class[c]["tn"] for c in CLASSES)
tot_fp = sum(per_class[c]["fp"] for c in CLASSES)
micro_spec = tot_tn / (tot_tn + tot_fp) if (tot_tn + tot_fp) else 0.0

# ===========================================================================
# 5. ROC-AUC AND PR-AUC FROM THE ACTUAL PROBABILITIES
# ===========================================================================
roc = {}
for i, c in enumerate(CLASSES):
    yt = Y_test[:, i]
    if len(np.unique(yt)) < 2:
        roc[c] = None
        continue
    fpr, tpr, _ = roc_curve(yt, Y_prob[:, i])
    roc[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}

valid_aucs = [roc[c]["auc"] for c in CLASSES if roc[c] is not None]
macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

fpr_micro, tpr_micro, _ = roc_curve(Y_test.ravel(), Y_prob.ravel())
micro_auc = float(auc(fpr_micro, tpr_micro))

pr = {}
for i, c in enumerate(CLASSES):
    yt = Y_test[:, i]
    if len(np.unique(yt)) < 2:
        pr[c] = None
        continue
    p, r, _ = precision_recall_curve(yt, Y_prob[:, i])
    ap = float(average_precision_score(yt, Y_prob[:, i]))
    pr[c] = {"precision": p, "recall": r, "ap": ap}

valid_aps = [pr[c]["ap"] for c in CLASSES if pr[c] is not None]
macro_ap = float(np.mean(valid_aps)) if valid_aps else None

# ===========================================================================
# 6. TEST LOSS (only what is legitimately derivable)
# ===========================================================================
# The training-time custom *weighted* BCE cannot be reproduced here because the
# class weights computed inside src/02_train_model.py were never saved.
# An unweighted BCE *is* an exact function of the saved probabilities and is
# therefore reported separately and clearly labelled.
eps = 1e-7
p_clip = np.clip(Y_prob.astype(float), eps, 1.0 - eps)
bce = float(-np.mean(Y_test * np.log(p_clip) + (1 - Y_test) * np.log(1 - p_clip)))

# ===========================================================================
# 7. VALIDATION METRICS
# ===========================================================================
val_block = {"available": False, "reason": NOT_ENOUGH}
if Y_val_prob is not None:
    Y_val_pred = (Y_val_prob >= thresholds).astype(int)
    val_block = {
        "available": True,
        "val_elementwise_accuracy": float(np.mean(Y_val_pred == Y_val)),
        "val_subset_accuracy": float(np.mean(np.all(Y_val_pred == Y_val, axis=1))),
        "val_micro_f1": float(f1_score(Y_val, Y_val_pred, average="micro", zero_division=0)),
        "val_macro_f1": float(f1_score(Y_val, Y_val_pred, average="macro", zero_division=0)),
    }
    log("")
    log("Validation (recomputed with the same thresholds):")
    log("  element-wise accuracy : %s" % pct(val_block["val_elementwise_accuracy"]))
    log("  subset accuracy       : %s" % pct(val_block["val_subset_accuracy"]))
    log("  micro F1              : %.4f" % val_block["val_micro_f1"])
else:
    log("")
    log("Validation accuracy: %s" % NOT_ENOUGH)
    log("  (src/03_evaluate_model.py printed it but never saved it to disk,\n"
        "   and no TensorFlow runtime is available here to recompute it.)")

# ===========================================================================
# 8. TRAINING HISTORY
# ===========================================================================
history_candidates = [
    os.path.join(RESULT_DIR, "history.json"),
    os.path.join(DATA_DIR, "history.json"),
    os.path.join(BASE_DIR, "history.json"),
    os.path.join(BASE_DIR, "training_history.json"),
]
history = None
for hp in history_candidates:
    if os.path.exists(hp):
        try:
            with open(hp, "r") as fh:
                history = json.load(fh)
            log("")
            log("Loaded real training history: %s" % hp)
            break
        except Exception:
            history = None

training_block = {"available": False, "reason": NOT_ENOUGH}
if history and ("loss" in history):
    key_acc = "binary_accuracy" if "binary_accuracy" in history else "accuracy"
    key_vacc = "val_binary_accuracy" if "val_binary_accuracy" in history else "val_accuracy"
    epochs = np.arange(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, history[key_acc], marker="o", ms=3.5, lw=1.6,
            color="#2166ac", label="Training accuracy")
    ax.plot(epochs, history[key_vacc], marker="s", ms=3.5, lw=1.6,
            color="#b2182b", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary accuracy")
    ax.set_xticks(epochs)
    ax.legend(frameon=False)
    save_fig(fig, "fig1_training_validation_accuracy.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, history["loss"], marker="o", ms=3.5, lw=1.6,
            color="#2166ac", label="Training loss")
    ax.plot(epochs, history["val_loss"], marker="s", ms=3.5, lw=1.6,
            color="#b2182b", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.legend(frameon=False)
    save_fig(fig, "fig2_training_validation_loss.png")

    training_block = {
        "available": True,
        "epochs_run": int(len(epochs)),
        "best_val_accuracy": float(np.max(history[key_vacc])),
        "best_epoch": int(np.argmax(history[key_vacc])) + 1,
        "final_train_accuracy": float(history[key_acc][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
    }
else:
    # No numeric History on disk. The real curves DO exist as images produced
    # by src/02_train_model.py - reuse them verbatim rather than inventing data.
    log("")
    log("No Keras History file found on disk (src/02_train_model.py saved only PNGs).")
    acc_png = os.path.join(RESULT_DIR, "improved_accuracy_curve.png")
    loss_png = os.path.join(RESULT_DIR, "improved_loss_curve.png")
    copied = []
    if os.path.exists(acc_png):
        shutil.copy2(acc_png, os.path.join(FIG_DIR, "fig1_training_validation_accuracy.png"))
        copied.append("fig1 <- results/improved_accuracy_curve.png")
    if os.path.exists(loss_png):
        shutil.copy2(loss_png, os.path.join(FIG_DIR, "fig2_training_validation_loss.png"))
        copied.append("fig2 <- results/improved_loss_curve.png")
    for c in copied:
        log("  reused existing real curve: %s" % c)
    if not copied:
        log("  no curve images found either -> %s" % NOT_ENOUGH)
    training_block = {
        "available": False,
        "reason": NOT_ENOUGH,
        "note": ("The trained model's curves exist only as the PNGs "
                 "results/improved_accuracy_curve.png and "
                 "results/improved_loss_curve.png, which are the genuine "
                 "training/validation plots from src/02_train_model.py. "
                 "Per-epoch numeric values were never saved."),
        "reused_images": bool(copied),
    }

# ===========================================================================
# 9. FIGURES FROM REAL PREDICTIONS
# ===========================================================================
log("")
log("Generating figures from the actual test predictions ...")

# --- Figure 3: one-vs-rest confusion matrices (native decision rule) -------
fig, axes = plt.subplots(1, N_CLASSES, figsize=(3.0 * N_CLASSES, 3.4))
if N_CLASSES == 1:
    axes = [axes]
for i, c in enumerate(CLASSES):
    cm = confusion_matrix(Y_test[:, i], Y_pred[:, i], labels=[0, 1])
    ax = axes[i]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(cm.max() if cm.max() else 1))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_yticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    if i == 0:
        ax.set_ylabel("Actual", fontsize=9)
    ax.set_title("%s\n(thr = %.2f)" % (c, thresholds[i]), fontsize=10)
    ax.grid(False)
    vmax = cm.max() if cm.max() else 1
    for r in range(2):
        for col in range(2):
            ax.text(col, r, "%d" % cm[r, col], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[r, col] > vmax / 2 else "black")
fig.suptitle("One-vs-Rest Confusion Matrices per Disease Class (test split)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
save_fig(fig, "fig3_confusion_matrix_ovr.png")

# --- Figure 4: precision / recall / F1 per class ---------------------------
x = np.arange(N_CLASSES)
w = 0.26
fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.bar(x - w, [per_class[c]["precision"] for c in CLASSES], w,
       label="Precision", color="#2166ac")
ax.bar(x, [per_class[c]["recall"] for c in CLASSES], w,
       label="Recall", color="#b2182b")
ax.bar(x + w, [per_class[c]["f1"] for c in CLASSES], w,
       label="F1-score", color="#1b7837")
for xi, c in zip(x, CLASSES):
    for off, key in ((-w, "precision"), (0.0, "recall"), (w, "f1")):
        ax.text(xi + off, per_class[c][key] + 0.015, "%.2f" % per_class[c][key],
                ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_xlabel("Disease class")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.10)
ax.legend(loc="upper right", frameon=False, ncol=3)
save_fig(fig, "fig4_classwise_precision_recall_f1.png")

# --- Figure 5: class distribution ------------------------------------------
train_pos = Y_train.sum(axis=0).astype(int)
val_pos = Y_val.sum(axis=0).astype(int)
test_pos = Y_test.sum(axis=0).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
axes[0].pie(test_pos, labels=["%s (%d)" % (c, v) for c, v in zip(CLASSES, test_pos)],
            autopct="%1.1f%%", startangle=90, colors=COLORS,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 9})
axes[0].set_title("Test split (N = %d records)" % len(Y_test), fontsize=11)
axes[0].grid(False)
axes[1].pie(train_pos, labels=["%s (%d)" % (c, v) for c, v in zip(CLASSES, train_pos)],
            autopct="%1.1f%%", startangle=90, colors=COLORS,
            wedgeprops={"edgecolor": "white", "linewidth": 1.0},
            textprops={"fontsize": 9})
axes[1].set_title("Training split (N = %d records)" % len(Y_train), fontsize=11)
axes[1].grid(False)
fig.tight_layout()
save_fig(fig, "fig5_class_distribution.png")

# --- Figure 6: ROC curves ---------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.8))
for i, c in enumerate(CLASSES):
    if roc[c] is not None:
        ax.plot(roc[c]["fpr"], roc[c]["tpr"], color=COLORS[i % len(COLORS)], lw=1.8,
                label="%s (AUC = %.3f)" % (c, roc[c]["auc"]))
ax.plot(fpr_micro, tpr_micro, color="black", linestyle=":", lw=1.7,
        label="Micro-average (AUC = %.3f)" % micro_auc)
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.0, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.02)
ax.legend(loc="lower right", frameon=False)
save_fig(fig, "fig6_roc_curves.png")

# --- Figure 7: precision-recall curves -------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.8))
for i, c in enumerate(CLASSES):
    if pr[c] is not None:
        ax.plot(pr[c]["recall"], pr[c]["precision"], color=COLORS[i % len(COLORS)], lw=1.8,
                label="%s (AP = %.3f)" % (c, pr[c]["ap"]))
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision")
ax.set_xlim(0.0, 1.02); ax.set_ylim(0.0, 1.02)
ax.legend(loc="lower left", frameon=False)
save_fig(fig, "fig7_precision_recall_curves.png")

# --- Figure 8: overall performance -----------------------------------------
names = ["Micro\nPrecision", "Micro\nRecall", "Micro\nF1",
         "Macro\nF1", "Weighted\nF1", "Macro\nROC-AUC",
         "Micro\nROC-AUC", "Test\nAccuracy"]
vals = [micro_p, micro_r, micro_f, macro_f, wei_f,
        macro_auc if macro_auc is not None else 0.0,
        micro_auc, element_acc]
fig, ax = plt.subplots(figsize=(9.0, 4.8))
bars = ax.bar(names, vals, color="#2166ac", width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, "%.3f" % v,
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.10)
save_fig(fig, "fig8_overall_performance.png")

# ===========================================================================
# 10. TABLES
# ===========================================================================
log("")
log("Writing tables ...")


def write_md_table(path, headers, rows, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s\n\n" % title)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")


# ---- Table 1: dataset / class distribution --------------------------------
rows1 = []
for i, c in enumerate(CLASSES):
    rows1.append([c, int(train_pos[i]), int(val_pos[i]), int(test_pos[i])])
rows1.append(["Total records", len(Y_train), len(Y_val), len(Y_test)])

with open(os.path.join(TAB_DIR, "table1_dataset_class_distribution.csv"), "w") as f:
    f.write("Class,Train positives,Val positives,Test positives\n")
    for r in rows1:
        f.write("%s,%d,%d,%d\n" % (r[0], r[1], r[2], r[3]))

write_md_table(
    os.path.join(TAB_DIR, "table1_dataset_class_distribution.md"),
    ["Class", "Train positives", "Val positives", "Test positives"],
    rows1,
    "Table 1. Distribution of disease categories across the PTB-XL splits.",
)

# ---- Table 2: classification report ---------------------------------------
rows2 = []
for c in CLASSES:
    d = per_class[c]
    rows2.append([c, "%.3f" % d["precision"], "%.3f" % d["recall"],
                  "%.3f" % d["f1"], d["support"]])
rows2.append(["Micro average", "%.3f" % micro_p, "%.3f" % micro_r,
              "%.3f" % micro_f, int(Y_test.size)])
rows2.append(["Macro average", "%.3f" % macro_p, "%.3f" % macro_r,
              "%.3f" % macro_f, int(Y_test.size)])
rows2.append(["Weighted average", "%.3f" % wei_p, "%.3f" % wei_r,
              "%.3f" % wei_f, int(Y_test.size)])

with open(os.path.join(TAB_DIR, "table2_classification_report.csv"), "w") as f:
    f.write("Class,Precision,Recall,F1-score,Support\n")
    for r in rows2:
        f.write("%s,%s,%s,%s,%s\n" % tuple(r))

write_md_table(
    os.path.join(TAB_DIR, "table2_classification_report.md"),
    ["Class", "Precision", "Recall", "F1-score", "Support"],
    rows2,
    "Table 2. Per-class classification report on the PTB-XL test split.",
)

# ---- Table 3: overall evaluation metrics ----------------------------------
def fmt(v):
    return "%.4f" % v if isinstance(v, float) else str(v)

rows3 = [
    ["Test accuracy (element-wise binary accuracy)", fmt(element_acc)],
    ["Subset accuracy (exact multi-label match)", fmt(subset_acc)],
    ["Hamming loss", fmt(hamming)],
    ["Precision (micro)", fmt(micro_p)],
    ["Precision (macro)", fmt(macro_p)],
    ["Precision (weighted)", fmt(wei_p)],
    ["Recall / Sensitivity (micro)", fmt(micro_r)],
    ["Recall / Sensitivity (macro)", fmt(macro_r)],
    ["Recall / Sensitivity (weighted)", fmt(wei_r)],
    ["Specificity (macro)", fmt(macro_spec)],
    ["Specificity (micro)", fmt(micro_spec)],
    ["F1-score (micro)", fmt(micro_f)],
    ["F1-score (macro)", fmt(macro_f)],
    ["F1-score (weighted)", fmt(wei_f)],
    ["ROC-AUC (macro)", fmt(macro_auc) if macro_auc is not None else NOT_ENOUGH],
    ["ROC-AUC (micro)", fmt(micro_auc)],
    ["PR-AUC / Average Precision (macro)", fmt(macro_ap) if macro_ap is not None else NOT_ENOUGH],
    ["Validation accuracy",
     fmt(val_block["val_elementwise_accuracy"]) if val_block.get("available") else NOT_ENOUGH],
    ["Training-time weighted BCE loss (test)", NOT_ENOUGH],
    ["Test binary cross-entropy (unweighted, from saved probabilities)", fmt(bce)],
]

with open(os.path.join(TAB_DIR, "table3_overall_evaluation_metrics.csv"), "w") as f:
    f.write("Metric,Value\n")
    for r in rows3:
        f.write("%s,%s\n" % (r[0], r[1]))

write_md_table(
    os.path.join(TAB_DIR, "table3_overall_evaluation_metrics.md"),
    ["Metric", "Value"],
    rows3,
    "Table 3. Overall evaluation metrics of the proposed model.",
)

# ===========================================================================
# 11. MACHINE-READABLE DUMP
# ===========================================================================
all_metrics = {
    "classes": CLASSES,
    "threshold_source": thr_source,
    "thresholds": [float(t) for t in thresholds],
    "probability_source": prob_source,
    "n_train": int(len(Y_train)),
    "n_val": int(len(Y_val)),
    "n_test": int(len(Y_test)),
    "train_positives": train_pos.tolist(),
    "val_positives": val_pos.tolist(),
    "test_positives": test_pos.tolist(),
    "per_class": per_class,
    "aggregate": {
        "test_elementwise_accuracy": element_acc,
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "weighted": {"precision": wei_p, "recall": wei_r, "f1": wei_f},
        "specificity_macro": macro_spec,
        "specificity_micro": micro_spec,
        "roc_auc_macro": macro_auc,
        "roc_auc_micro": micro_auc,
        "roc_auc_per_class": {c: (roc[c]["auc"] if roc[c] is not None else None) for c in CLASSES},
        "pr_auc_macro": macro_ap,
        "pr_auc_per_class": {c: (pr[c]["ap"] if pr[c] is not None else None) for c in CLASSES},
        "bce_unweighted": bce,
    },
    "validation": val_block,
    "training_history": training_block,
    "not_available": {
        "training_time_weighted_bce_test_loss": NOT_ENOUGH,
        "per_epoch_training_history": NOT_ENOUGH if not training_block.get("available") else False,
    },
}

with open(os.path.join(TAB_DIR, "metrics_all.json"), "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

# ===========================================================================
# 12. CONSOLE SUMMARY (copy-paste friendly)
# ===========================================================================
log("")
log("=" * 74)
log(" RESULTS SUMMARY")
log("=" * 74)
log("Test accuracy (element-wise) : %s" % pct(element_acc))
log("Subset accuracy              : %s" % pct(subset_acc))
log("Hamming loss                 : %.4f" % hamming)
log("Micro  P/R/F1                : %.4f / %.4f / %.4f" % (micro_p, micro_r, micro_f))
log("Macro  P/R/F1                : %.4f / %.4f / %.4f" % (macro_p, macro_r, macro_f))
log("Weighted P/R/F1              : %.4f / %.4f / %.4f" % (wei_p, wei_r, wei_f))
log("Specificity (macro / micro)  : %.4f / %.4f" % (macro_spec, micro_spec))
log("ROC-AUC (macro / micro)      : %s / %.4f" %
    ("%.4f" % macro_auc if macro_auc is not None else NOT_ENOUGH, micro_auc))
log("PR-AUC (macro)               : %s" %
    ("%.4f" % macro_ap if macro_ap is not None else NOT_ENOUGH))
log("Test BCE (unweighted)        : %.4f" % bce)
log("")
log("Per-class:")
for c in CLASSES:
    d = per_class[c]
    log("  %-5s P=%.3f R=%.3f F1=%.3f Spec=%.3f Support=%d (TP=%d FP=%d FN=%d TN=%d)" %
        (c, d["precision"], d["recall"], d["f1"], d["specificity"],
         d["support"], d["tp"], d["fp"], d["fn"], d["tn"]))
log("")
log("Output written to: results_paper/figures , results_paper/tables , metrics_all.json")
log("=" * 74)
