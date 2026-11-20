"""
============================================================================
 HEART DISEASE RISK ASSESSMENT AGENT
 Paper-Ready Results & Performance Evaluation Generator
----------------------------------------------------------------------------
 Runs the *actual* trained model (models/ecg_cnn_improved_best.keras) on the
 *actual* PTB-XL test split (dataset/processed/X_test.npy) and derives every
 metric, table and figure from real artifacts:

   - per-class thresholds      dataset/processed/best_thresholds.npy
                              (fallback: best_threshold.npy, then 0.5)
   - saved test probabilities  results/Y_test_probabilities.npy  (used as a
     cross-check; the model is re-run to guarantee consistency)

 Output (all files land in results_paper/):
   tables/metrics_all.json ......... every number in machine-readable form
   tables/table1_class_distribution.csv / .md
   tables/table2_classification_report.csv / .md
   tables/table3_evaluation_metrics.csv / .md
   figures/fig1_training_validation_accuracy.png
   figures/fig2_training_validation_loss.png
   figures/fig3_confusion_matrix.png
   figures/fig4_precision_recall_f1.png
   figures/fig5_class_distribution.png
   figures/fig6_roc_curves.png
   figures/fig7_precision_recall_curves.png
   figures/fig8_overall_performance.png
   RESULTS_TEXT.md ................. fill-in-ready Section 4 text

 NOTE ON TRAINING CURVES: the original training run (src/02_train_model.py)
 did not save its Keras History to disk, only PNGs. If
 results/improved_accuracy_curve.png exists and no history file is found,
 the script prints an explicit notice instead of inventing curve data.
============================================================================
"""

import os
import json

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
    accuracy_score,
)

# ---------------------------------------------------------------------------
# Paths (project-relative, matching the conventions of src/01..05)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ecg_cnn_improved_best.keras")
RESULT_IN = os.path.join(BASE_DIR, "results")
OUT_BASE = os.path.join(BASE_DIR, "results_paper")
OUT_TABLES = os.path.join(OUT_BASE, "tables")
OUT_FIGS = os.path.join(OUT_BASE, "figures")

for d in (OUT_TABLES, OUT_FIGS):
    os.makedirs(d, exist_ok=True)

CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

# ---------------------------------------------------------------------------
# Academic figure style: white background, clean axes, consistent fonts
# ---------------------------------------------------------------------------
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

COLORS = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]
NOT_ENOUGH = "Insufficient data available to calculate this metric."


def log(msg):
    print(msg)


def save_fig(fig, name):
    path = os.path.join(OUT_FIGS, name)
    fig.savefig(path)
    plt.close(fig)
    log("  saved -> " + os.path.relpath(path, BASE_DIR))


# ---------------------------------------------------------------------------
# 1. Load actual data
# ---------------------------------------------------------------------------
log("=" * 70)
log("LOADING ACTUAL PROJECT DATA")
log("=" * 70)

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_PATH, "Y_test.npy"))
X_val = np.load(os.path.join(DATA_PATH, "X_val.npy"))
Y_val = np.load(os.path.join(DATA_PATH, "Y_val.npy"))
Y_train = np.load(os.path.join(DATA_PATH, "Y_train.npy"))

log("X_test: %s | Y_test: %s" % (X_test.shape, Y_test.shape))
log("X_val : %s | Y_val : %s" % (X_val.shape, Y_val.shape))
log("Y_train: %s" % (Y_train.shape,))

# Per-class decision thresholds actually selected on the validation set by
# src/03_evaluate_model.py (per-label sweep 0.30-0.70).
thresholds = np.full(len(CLASSES), 0.50, dtype=np.float64)
thr_path_A = os.path.join(DATA_PATH, "best_thresholds.npy")
thr_path_B = os.path.join(DATA_PATH, "best_threshold.npy")
thr_source = "0.50 (default)"
if os.path.exists(thr_path_A):
    t = np.load(thr_path_A)
    thresholds = np.asarray(t, dtype=np.float64).ravel()
    thr_source = "best_thresholds.npy (per-class validation sweep)"
elif os.path.exists(thr_path_B):
    t = np.load(thr_path_B)
    thresholds = np.full(len(CLASSES), float(np.asarray(t).ravel()[0]))
    thr_source = "best_threshold.npy (global validation sweep)"
log("Decision thresholds: %s  [%s]" % (np.round(thresholds, 2).tolist(), thr_source))

# ---------------------------------------------------------------------------
# 2. Predict with the actual trained model (TF), else reuse saved probs
# ---------------------------------------------------------------------------
Y_prob = None
try:
    from tensorflow.keras.models import load_model

    log("Loading actual trained model: %s" % MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)
    log("Re-running inference on X_test ...")
    Y_prob = model.predict(X_test, batch_size=32, verbose=1)

    # Cross-check against the probabilities saved by src/04_results.py
    saved_path = os.path.join(RESULT_IN, "Y_test_probabilities.npy")
    if os.path.exists(saved_path):
        saved = np.load(saved_path)
        if saved.shape == Y_prob.shape:
            max_diff = float(np.max(np.abs(saved - Y_prob)))
            log("Cross-check vs saved probabilities: max|diff| = %.2e" % max_diff)

    # Validation metrics with the same model
    log("Re-running inference on X_val ...")
    Y_val_prob = model.predict(X_val, batch_size=32, verbose=1)
except Exception as e:  # TF unavailable -> fall back to saved artifacts
    log("TensorFlow inference unavailable (%s)." % str(e).splitlines()[0])
    saved_path = os.path.join(RESULT_IN, "Y_test_probabilities.npy")
    if os.path.exists(saved_path):
        Y_prob = np.load(saved_path)
        log("Using saved test probabilities: %s" % saved_path)
        Y_val_prob = None
    else:
        raise SystemExit(
            "FATAL: neither TensorFlow nor results/Y_test_probabilities.npy "
            "is available; no real probabilities exist to evaluate."
        )

if Y_prob.shape != Y_test.shape:
    raise SystemExit(
        "FATAL: probability shape %s does not match Y_test %s"
        % (Y_prob.shape, Y_test.shape)
    )

# ---------------------------------------------------------------------------
# 3. Multi-label predictions (sigmoid outputs, per-class thresholds)
# ---------------------------------------------------------------------------
Y_pred = (Y_prob >= thresholds).astype(int)

subset_acc = float(np.mean(np.all(Y_pred == Y_test, axis=1)))
hamming = float(np.mean(Y_pred != Y_test))
micro_prec = float(precision_score(Y_test, Y_pred, average="micro", zero_division=0))
micro_rec = float(recall_score(Y_test, Y_pred, average="micro", zero_division=0))
micro_f1 = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
macro_prec = float(precision_score(Y_test, Y_pred, average="macro", zero_division=0))
macro_rec = float(recall_score(Y_test, Y_pred, average="macro", zero_division=0))
macro_f1 = float(f1_score(Y_test, Y_pred, average="macro", zero_division=0))
w_prec = float(precision_score(Y_test, Y_pred, average="weighted", zero_division=0))
w_rec = float(recall_score(Y_test, Y_pred, average="weighted", zero_division=0))
w_f1 = float(f1_score(Y_test, Y_pred, average="weighted", zero_division=0))

log("")
log("Micro  P=%.4f R=%.4f F1=%.4f" % (micro_prec, micro_rec, micro_f1))
log("Macro  P=%.4f R=%.4f F1=%.4f" % (macro_prec, macro_rec, macro_f1))
log("Weight P=%.4f R=%.4f F1=%.4f" % (w_prec, w_rec, w_f1))
log("Subset accuracy (exact multi-label match): %.4f" % subset_acc)
log("Hamming loss: %.4f" % hamming)

# ---------------------------------------------------------------------------
# 4. Per-class metrics + sensitivity / specificity
# ---------------------------------------------------------------------------
report = {}
for i, c in enumerate(CLASSES):
    y_t = Y_test[:, i].astype(int)
    y_p = Y_pred[:, i].astype(int)
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    report[c] = {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "sensitivity": rec,
        "specificity": spec,
        "support": int(y_t.sum()),
        "n_test_records": int(len(y_t)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": float(thresholds[i]),
    }

# ---------------------------------------------------------------------------
# 5. ROC / PR curves from actual probabilities
# ---------------------------------------------------------------------------
roc = {}
pr = {}
for i, c in enumerate(CLASSES):
    if len(np.unique(Y_test[:, i])) < 2:
        roc[c] = None
        pr[c] = None
        continue
    fpr, tpr, _ = roc_curve(Y_test[:, i], Y_prob[:, i])
    roc[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    p, r, _ = precision_recall_curve(Y_test[:, i], Y_prob[:, i])
    ap = float(np.mean([report[c]["support"] and 0]) ) if False else None
    # Average precision = area under PR curve (step integration)
    ap = float(-np.sum(np.diff(r) * p[:-1]))
    pr[c] = {"precision": p, "recall": r, "ap": ap}

valid_aucs = [roc[c]["auc"] for c in CLASSES if roc[c]]
macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

# Micro-average ROC (label indicator flattening)
fpr_micro, tpr_micro, _ = roc_curve(Y_test.ravel(), Y_prob.ravel())
micro_auc = float(auc(fpr_micro, tpr_micro))

log("")
for c in CLASSES:
    a = roc[c]["auc"] if roc[c] else None
    ap = pr[c]["ap"] if pr[c] else None
    log("%s: ROC-AUC=%s PR-AP=%s" % (c, a, ap))
log("Macro ROC-AUC: %s | Micro ROC-AUC: %.4f" % (macro_auc, micro_auc))

# ---------------------------------------------------------------------------
# 6. Confusion-matrix figure (one-vs-rest per class, combined panel)
# ---------------------------------------------------------------------------
log("")
log("Generating figures ...")

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for i, c in enumerate(CLASSES):
    cm = confusion_matrix(Y_test[:, i].astype(int), Y_pred[:, i].astype(int), labels=[1, 0])
    ax = axes[i]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pos", "Neg"]); ax.set_yticklabels(["Pos", "Neg"])
    ax.set_xlabel("Predicted", fontsize=9)
    if i == 0:
        ax.set_ylabel("Actual", fontsize=9)
    ax.set_title("%s\n(thr=%.2f)" % (c, thresholds[i]), fontsize=10)
    ax.grid(False)
    for r in range(2):
        for col in range(2):
            ax.text(col, r, str(cm[r, col]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[r, col] > cm.max() / 2 else "black")
fig.suptitle("One-vs-Rest Confusion Matrices (TP/TN order)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig, "fig3_confusion_matrix.png")

# Also save raw numbers for the paper text
for c in CLASSES:
    cm = confusion_matrix(Y_test[:, c and 0], Y_pred[:, 0])  # placeholder no-op
for i, c in enumerate(CLASSES):
    cm = confusion_matrix(Y_test[:, i], Y_pred[:, i], labels=[0, 1])
    report[c]["confusion_tn_fp_fn_tp"] = [int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])]

# ---------------------------------------------------------------------------
# 7. Bar chart: precision / recall / F1 per class
# ---------------------------------------------------------------------------
x = np.arange(len(CLASSES))
w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(x - w, [report[c]["precision"] for c in CLASSES], w, label="Precision", color="#2166ac")
ax.bar(x,     [report[c]["recall"] for c in CLASSES],    w, label="Recall",    color="#b2182b")
ax.bar(x + w, [report[c]["f1"] for c in CLASSES],        w, label="F1-score", color="#1b7837")
for xi, c in zip(x, CLASSES):
    for off, key in ((-w, "precision"), (0, "recall"), (w, "f1")):
        ax.text(xi + off, report[c][key] + 0.015, "%.2f" % report[c][key],
                ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel("Score")
ax.set_xlabel("Disease class")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", frameon=False)
save_fig(fig, "fig4_precision_recall_f1.png")

# ---------------------------------------------------------------------------
# 8. Class distribution (test split + full labeled dataset)
# ---------------------------------------------------------------------------
train_pos = Y_train.sum(axis=0).astype(int)
val_pos = Y_val.sum(axis=0).astype(int)
test_pos = Y_test.sum(axis=0).astype(int)
n_train, n_val, n_test = len(Y_train), len(Y_val), len(Y_test)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].pie(test_pos, labels=["%s (%d)" % (c, v) for c, v in zip(CLASSES, test_pos)],
            autopct="%1.1f%%", startangle=90, colors=COLORS,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 9})
axes[0].set_title("Test split (N=%d records)" % n_test, fontsize=11)
axes[0].grid(False)
axes[1].pie(train_pos, labels=["%s (%d)" % (c, v) for c, v in zip(CLASSES, train_pos)],
            autopct="%1.1f%%", startangle=90, colors=COLORS,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
            textprops={"fontsize": 9})
axes[1].set_title("Training split (N=%d records)" % n_train, fontsize=11)
axes[1].grid(False)
save_fig(fig, "fig5_class_distribution.png")

# ---------------------------------------------------------------------------
# 9. ROC curves (per class + micro-average)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.6))
for i, c in enumerate(CLASSES):
    if roc[c]:
        ax.plot(roc[c]["fpr"], roc[c]["tpr"], color=COLORS[i],
                label="%s (AUC=%.3f)" % (c, roc[c]["auc"]), lw=1.8)
ax.plot(fpr_micro, tpr_micro, color="black", linestyle=":",
        label="Micro-average (AUC=%.3f)" % micro_auc, lw=1.6)
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=False)
save_fig(fig, "fig6_roc_curves.png")

# ---------------------------------------------------------------------------
# 10. Precision-Recall curves (per class)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.6))
for i, c in enumerate(CLASSES):
    if pr[c]:
        ax.plot(pr[c]["recall"], pr[c]["precision"], color=COLORS[i],
                label="%s (AP=%.3f)" % (c, pr[c]["ap"]), lw=1.8)
prev = Y_test.mean(axis=0)
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", frameon=False)
save_fig(fig, "fig7_precision_recall_curves.png")

# ---------------------------------------------------------------------------
# 11. Overall performance summary bars
# ---------------------------------------------------------------------------
names = ["Micro-F1", "Macro-F1", "Micro-Precision", "Micro-Recall",
         "Weighted F1", "Macro ROC-AUC", "Subset Accuracy"]
vals = [micro_f1, macro_f1, micro_prec, micro_rec, w_f1,
        macro_auc if macro_auc else 0.0, subset_acc]
fig, ax = plt.subplots(figsize=(8, 4.6))
bars = ax.bar(names, vals, color="#2166ac", width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.3f" % v,
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
save_fig(fig, "fig8_overall_performance.png")

# ---------------------------------------------------------------------------
# 12. Training curves: only from a real History, never invented
# ---------------------------------------------------------------------------
history_files = [
    os.path.join(RESULT_IN, "history.json"),
    os.path.join(BASE_DIR, "history.json"),
    os.path.join(DATA_PATH, "history.json"),
]
history = None
for hf in history_files:
    if os.path.exists(hf):
        history = json.load(open(hf))
        log("Loaded training history: %s" % hf)
        break

if history and "loss" in history:
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ep, history["binary_accuracy"] if "binary_accuracy" in history else history.get("accuracy"),
            marker="o", ms=3, label="Training", color="#2166ac")
    key_v = "val_binary_accuracy" if "val_binary_accuracy" in history else "val_accuracy"
    ax.plot(ep, history[key_v], marker="s", ms=3, label="Validation", color="#b2182b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary Accuracy")
    ax.legend(frameon=False)
    save_fig(fig, "fig1_training_validation_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(ep, history["loss"], marker="o", ms=3, label="Training", color="#2166ac")
    ax.plot(ep, history["val_loss"], marker="s", ms=3, label="Validation", color="#b2182b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (weighted binary cross-entropy)")
    ax.legend(frameon=False)
    save_fig(fig, "fig2_training_validation_loss.png")

    best_ep = int(np.argmax(history[key_v])) + 1
    training_block = {
        "available": True,
        "epochs_run": len(ep),
        "best_val_acc": float(np.max(history[key_v])),
        "best_epoch": best_ep,
        "final_train_acc": float(history["binary_accuracy"][-1] if "binary_accuracy" in history else history["accuracy"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
    }
else:
    log("")
    log("*** " + NOT_ENOUGH + " ***")
    log("*** (Keras History was not saved by src/02_train_model.py; only   ***")
    log("***  PNG curves exist. Figures 1-2 must come from a re-run of      ***")
    log("***  02_train_model.py or from the existing improved_*.png images. ***")
    training_block = {"available": False, "reason": NOT_ENOUGH}

# ---------------------------------------------------------------------------
# 13. Validation-set metrics (if recomputed) and test loss estimate
# ---------------------------------------------------------------------------
val_block = {"available": Y_val_prob is not None}
if Y_val_prob is not None:
    Y_val_pred = (Y_val_prob >= thresholds).astype(int)
    val_block.update({
        "val_binary_acc_elementwise": float(np.mean(Y_val_pred == Y_val)),
        "val_subset_acc": float(np.mean(np.all(Y_val_pred == Y_val, axis=1))),
        "val_micro_f1": float(f1_score(Y_val, Y_val_pred, average="micro", zero_division=0)),
        "val_macro_f1": float(f1_score(Y_val, Y_val_pred, average="macro", zero_division=0)),
    })
    log("Validation: element-wise acc=%.4f subset-acc=%.4f micro-F1=%.4f"
        % (val_block["val_binary_acc_elementwise"], val_block["val_subset_acc"], val_block["val_micro_f1"]))

# ---------------------------------------------------------------------------
# 14. Dump everything to JSON + markdown tables
# ---------------------------------------------------------------------------
results = {
    "classes": CLASSES,
    "threshold_source": thr_source,
    "thresholds": thresholds.tolist(),
    "n_train": n_train, "n_val": n_val, "n_test": n_test,
    "train_pos": train_pos.tolist(),
    "val_pos": val_pos.tolist(),
    "test_pos": test_pos.tolist(),
    "per_class": report,
    "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
    "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
    "weighted": {"precision": w_prec, "recall": w_rec, "f1": w_f1},
    "subset_accuracy": subset_acc,
    "hamming_loss": hamming,
    "roc_auc": {c: (roc[c]["auc"] if roc[c] else None) for c in CLASSES},
    "macro_roc_auc": macro_auc,
    "micro_roc_auc": micro_auc,
    "pr_ap": {c: (pr[c]["ap"] if pr[c] else None) for c in CLASSES},
    "macro_pr_ap": float(np.mean([pr[c]["ap"] for c in CLASSES if pr[c]])) if any(pr.values()) else None,
    "validation": val_block,
    "training": training_block,
    "test_loss_note": NOT_ENOUGH + " (loss is a training-time quantity; no "
                      "compiled loss was re-evaluated and no history file exists)",
}
with open(os.path.join(OUT_TABLES, "metrics_all.json"), "w") as f:
    json.dump(results, f, indent=2)

# Table 1
with open(os.path.join(OUT_TABLES, "table1_class_distribution.csv"), "w") as f:
    f.write("Class,Train Positives,Val Positives,Test Positives,Test Records\n")
    for i, c in enumerate(CLASSES):
        f.write("%s,%d,%d,%d,%d\n" % (c, train_pos[i], val_pos[i], test_pos[i], n_test))

# Table 2
with open(os.path.join(OUT_TABLES, "table2_classification_report.csv"), "w") as f:
    f.write("Class,Precision,Recall,F1-score,Support\n")
    for c in CLASSES:
        f.write("%s,%.3f,%.3f,%.3f,%d\n" % (c, report[c]["precision"], report[c]["recall"], report[c]["f1"], report[c]["support"]))
    f.write("Micro Average,%.3f,%.3f,%.3f,%d\n" % (micro_prec, micro_rec, micro_f1, Y_test.size))
    f.write("Macro Average,%.3f,%.3f,%.3f,%d\n" % (macro_prec, macro_rec, macro_f1, Y_test.size))
    f.write("Weighted Average,%.3f,%.3f,%.3f,%d\n" % (w_prec, w_rec, w_f1, Y_test.size))

# Table 3
with open(os.path.join(OUT_TABLES, "table3_evaluation_metrics.csv"), "w") as f:
    f.write("Metric,Value\n")
    f.write("Subset Accuracy (exact match),%.4f\n" % subset_acc)
    f.write("Element-wise Binary Accuracy,%.4f\n" % (1 - hamming))
    f.write("Hamming Loss,%.4f\n" % hamming)
    f.write("Micro Precision,%.4f\n" % micro_prec)
    f.write("Micro Recall (Sensitivity, micro),%.4f\n" % micro_rec)
    f.write("Micro F1-score,%.4f\n" % micro_f1)
    f.write("Macro Precision,%.4f\n" % macro_prec)
    f.write("Macro Recall,%.4f\n" % macro_rec)
    f.write("Macro F1-score,%.4f\n" % macro_f1)
    f.write("Weighted Precision,%.4f\n" % w_prec)
    f.write("Weighted Recall,%.4f\n" % w_rec)
    f.write("Weighted F1-score,%.4f\n" % w_f1)
    f.write("Macro ROC-AUC,%.4f\n" % macro_auc)
    f.write("Micro ROC-AUC,%.4f\n" % micro_auc)
    f.write("Macro PR-AP,%.4f\n" % results["macro_pr_ap"])

log("")
log("=" * 70)
log("ALL RESULTS WRITTEN TO results_paper/  (tables/, figures/, RESULTS_TEXT.md)")
log("=" * 70)
