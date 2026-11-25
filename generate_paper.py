"""
generate_paper.py  -  Heart Disease Risk Assessment Agent
Generates all figures (PNG, 300 dpi), tables (CSV) and a Word research paper (.docx)
using ONLY the actual project results supplied by the author.

Run from the project root:
    pip install matplotlib numpy pandas python-docx
    python generate_paper.py

Training history (Figures 1 and 2):
  The script searches the project for a saved history file (JSON/CSV, e.g. history.json,
  training_history.csv, history.csv). If none is found, Figures 1-2 are SKIPPED.
  No epoch values are ever invented.

Model architecture (Table 2):
  The script tries to load a saved Keras model (*.keras / *.h5) and reads its real layers.
  If none is found, Table 2 is left as a clearly marked placeholder for you to fill.
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = "paper_output"
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
for d in (FIG, TAB):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. ACTUAL PROJECT RESULTS (as supplied - do not edit unless project changes)
# ----------------------------------------------------------------------------
CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
OVERALL = {"Training Accuracy": 91.07, "Validation Accuracy": 88.12,
           "Test Binary Accuracy": 87.58, "Mean ROC-AUC": 90.52}
CLASSWISE = {  # accuracy %, precision, recall, F1, ROC-AUC
    "NORM": (85.29, 0.7950, 0.9138, 0.8503, 0.9382),
    "MI":   (85.81, 0.7707, 0.6110, 0.6816, 0.9168),
    "STTC": (88.60, 0.7679, 0.7629, 0.7654, 0.9305),
    "HYP":  (88.81, 0.6216, 0.2875, 0.3932, 0.8225),
    "CD":   (89.39, 0.8333, 0.6517, 0.7314, 0.9178),
}
CM = {  # TN, FP, FN, TP
    "NORM": (828, 205, 75, 795),
    "MI":   (1344, 86, 184, 289),
    "STTC": (1332, 107, 110, 354),
    "HYP":  (1621, 42, 171, 69),
    "CD":   (1426, 55, 147, 275),
}

# ----------------------------------------------------------------------------
# 2. Matplotlib IEEE-like style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
DPI = 300
GRAY = ["#222222", "#666666", "#aaaaaa"]

def save(fig, name):
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

# ----------------------------------------------------------------------------
# 3. Locate real training history (Figures 1-2)
# ----------------------------------------------------------------------------
def find_history():
    patterns = ["**/*history*.json", "**/*history*.csv", "**/*log*.csv"]
    for p in patterns:
        for f in glob.glob(p, recursive=True):
            if OUT in f or "node_modules" in f or "venv" in f:
                continue
            try:
                if f.endswith(".json"):
                    with open(f) as fh:
                        h = pd.DataFrame(json.load(fh))
                else:
                    h = pd.read_csv(f)
                cols = [c.lower() for c in h.columns]
                if any("loss" in c for c in cols):
                    h.columns = cols
                    return h, f
            except Exception:
                continue
    return None, None

def pick(h, *names):
    for n in names:
        if n in h.columns:
            return n
    return None

hist, hist_file = find_history()
figs = {}
if hist is not None:
    acc = pick(hist, "accuracy", "binary_accuracy", "acc")
    vacc = pick(hist, "val_accuracy", "val_binary_accuracy", "val_acc")
    loss, vloss = pick(hist, "loss"), pick(hist, "val_loss")
    ep = np.arange(1, len(hist) + 1)
    if acc and vacc:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(ep, hist[acc], color=GRAY[0], label="Training")
        ax.plot(ep, hist[vacc], color=GRAY[1], ls="--", label="Validation")
        ax.set(xlabel="Epoch", ylabel="Accuracy"); ax.legend()
        figs[1] = save(fig, "fig1_accuracy_curve.png")
    if loss and vloss:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(ep, hist[loss], color=GRAY[0], label="Training")
        ax.plot(ep, hist[vloss], color=GRAY[1], ls="--", label="Validation")
        ax.set(xlabel="Epoch", ylabel="Loss"); ax.legend()
        figs[2] = save(fig, "fig2_loss_curve.png")
    print("Training history used from:", hist_file)
else:
    print("WARNING: no training history file found -> Figures 1 and 2 skipped (no invented data).")

# ----------------------------------------------------------------------------
# 4. Figures 3-7 (actual values)
# ----------------------------------------------------------------------------
x = np.arange(len(CLASSES)); w = 0.26

# Fig 3 - Precision / Recall / F1
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for i, (lab, idx) in enumerate([("Precision", 1), ("Recall", 2), ("F1 Score", 3)]):
    vals = [CLASSWISE[c][idx] for c in CLASSES]
    b = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=GRAY[i], edgecolor="black", lw=0.5)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set(xlabel="ECG class", ylabel="Score", ylim=(0, 1.15)); ax.legend(ncol=3, loc="upper center")
figs[3] = save(fig, "fig3_precision_recall_f1.png")

# Fig 4 - ROC-AUC
fig, ax = plt.subplots(figsize=(3.5, 2.6))
vals = [CLASSWISE[c][4] for c in CLASSES]
b = ax.bar(CLASSES, vals, color=GRAY[1], edgecolor="black", lw=0.5)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
ax.set(xlabel="ECG class", ylabel="ROC-AUC", ylim=(0.5, 1.0))
figs[4] = save(fig, "fig4_classwise_rocauc.png")

# Fig 5 - Multilabel confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
for ax, c in zip(axes.flat, CLASSES):
    tn, fp, fn, tp = CM[c]
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap="Greys", vmin=0, vmax=m.max() * 1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > m.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(c); ax.grid(False)
axes.flat[-1].axis("off")
figs[5] = save(fig, "fig5_multilabel_confusion_matrix.png")

# Fig 6 - FP vs FN
fig, ax = plt.subplots(figsize=(3.5, 2.6))
fp = [CM[c][1] for c in CLASSES]; fn = [CM[c][2] for c in CLASSES]
b1 = ax.bar(x - 0.2, fp, 0.4, label="False Positives", color=GRAY[1], edgecolor="black", lw=0.5)
b2 = ax.bar(x + 0.2, fn, 0.4, label="False Negatives", color=GRAY[2], edgecolor="black", lw=0.5, hatch="//")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 3, str(int(r.get_height())), ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set(xlabel="ECG class", ylabel="Number of samples"); ax.legend()
figs[6] = save(fig, "fig6_fp_vs_fn.png")

# Fig 7 - Overall performance
fig, ax = plt.subplots(figsize=(3.5, 2.6))
labels = ["Train\nAcc.", "Val.\nAcc.", "Test\nAcc.", "Mean\nROC-AUC"]
vals = list(OVERALL.values())
b = ax.bar(labels, vals, color=GRAY[1], edgecolor="black", lw=0.5)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center", fontsize=8)
ax.set(ylabel="Percentage (%)", ylim=(80, 95))
figs[7] = save(fig, "fig7_overall_performance.png")

# ----------------------------------------------------------------------------
# 5. Tables
# ----------------------------------------------------------------------------
tables = {}
tables["Table 1 - Class information"] = pd.DataFrame({
    "Class": CLASSES,
    "Description": ["Normal ECG", "Myocardial Infarction", "ST/T Change",
                    "Hypertrophy", "Conduction Disturbance"],
    "Positive samples (test)": [CM[c][2] + CM[c][3] for c in CLASSES],
    "Negative samples (test)": [CM[c][0] + CM[c][1] for c in CLASSES],
})

# Table 2 - real layers from saved model if possible
arch_rows = None
# Prioritize ecg_cnn_improved_best.keras or find models
model_candidates = ["models/ecg_cnn_improved_best.keras", "models/ecg_cnn_best.keras", "models/ecg_cnn.keras"]
model_files = [f for f in model_candidates if os.path.exists(f)] + glob.glob("**/*.keras", recursive=True) + glob.glob("**/*.h5", recursive=True)
for f in model_files:
    try:
        from tensorflow import keras
        model = keras.models.load_model(f, compile=False)
        arch_rows = [(l.name, l.__class__.__name__, str(l.output.shape) if hasattr(l, "output") else "-",
                      l.count_params()) for l in model.layers]
        print("Architecture read from:", f)
        break
    except Exception as e:
        print("Could not load", f, "->", e)
if arch_rows:
    tables["Table 2 - Model architecture"] = pd.DataFrame(
        arch_rows, columns=["Layer", "Type", "Output shape", "Parameters"])
else:
    tables["Table 2 - Model architecture"] = pd.DataFrame(
        {"Layer": ["<FILL FROM model.summary()>"], "Type": ["-"], "Output shape": ["-"], "Parameters": ["-"]})

tables["Table 3 - Overall metrics"] = pd.DataFrame({
    "Metric": ["Training Accuracy", "Validation Accuracy", "Test Binary Accuracy", "Mean ROC-AUC"],
    "Value": ["91.07%", "88.12%", "87.58%", "0.9052"]})

tables["Table 4 - Class-wise metrics"] = pd.DataFrame(
    [[c, f"{v[0]:.2f}%", f"{v[1]:.4f}", f"{v[2]:.4f}", f"{v[3]:.4f}", f"{v[4]:.4f}"]
     for c, v in CLASSWISE.items()],
    columns=["Class", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])

tables["Table 5 - Confusion matrix values"] = pd.DataFrame(
    [[c, *CM[c]] for c in CLASSES], columns=["Class", "TN", "FP", "FN", "TP"])

err = []
for c in CLASSES:
    tn, fp_, fn_, tp = CM[c]
    n = tn + fp_ + fn_ + tp
    err.append([c, fp_, fn_, f"{fp_/(fp_+tn)*100:.2f}%", f"{fn_/(fn_+tp)*100:.2f}%",
                f"{(fp_+fn_)/n*100:.2f}%"])
tables["Table 6 - Error analysis"] = pd.DataFrame(
    err, columns=["Class", "FP", "FN", "False Positive Rate", "False Negative Rate", "Total Error"])

for name, df in tables.items():
    df.to_csv(os.path.join(TAB, name.split(" - ")[0].replace(" ", "_") + ".csv"), index=False)

# ----------------------------------------------------------------------------
# 6. Word document
# ----------------------------------------------------------------------------
doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)

def H(t): doc.add_heading(t, level=1)
def P(t): p = doc.add_paragraph(t); p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; return p
def FIG_(n, cap, width=3.4):
    if n in figs:
        doc.add_picture(figs[n], width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(f"Fig. {n}. {cap}"); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph(f"[Fig. {n} skipped: {cap} - training history file not found.]")
def TBL(key, cap):
    df = tables[key]
    doc.add_paragraph(f"TABLE {key.split()[1]}. {cap}").alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
        for r in t.rows[0].cells[i].paragraphs[0].runs: r.bold = True
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = str(v)
    doc.add_paragraph()

TODO = "[FILL FROM PROJECT FILES: %s]"
doc.add_heading("Heart Disease Risk Assessment Agent: An AI-Assisted System Combining Patient Health "
                "Parameters and CNN-Based ECG Analysis", 0)

H("Abstract")
P("This paper presents the Heart Disease Risk Assessment Agent, an AI-assisted research and educational "
  "system that combines patient health information with electrocardiogram (ECG) signal analysis. The ECG "
  "component uses a convolutional neural network (CNN) for multilabel classification of five diagnostic "
  "classes: NORM, MI, STTC, HYP and CD. The current implementation achieved a training accuracy of 91.07%, "
  "a validation accuracy of 88.12%, a test binary accuracy of 87.58% and a mean ROC-AUC of 0.9052. "
  "Class-wise ROC-AUC values ranged from 0.8225 (HYP) to 0.9382 (NORM). HYP showed the lowest recall "
  "(0.2875) and F1 score (0.3932), identifying it as the main area for improvement. The system is not a "
  "clinical diagnostic tool and does not replace professional medical evaluation.")
H("Keywords")
P("Heart disease risk assessment, ECG classification, convolutional neural network, multilabel "
  "classification, deep learning, ROC-AUC, AI-assisted healthcare.")
H("I. Introduction")
P("Cardiovascular diseases remain a leading cause of mortality worldwide, and the ECG is a widely used, "
  "non-invasive tool for assessing cardiac electrical activity. Deep learning methods, particularly CNNs, "
  "can learn discriminative features directly from raw ECG signals. This work describes an AI-assisted "
  "risk assessment system that combines patient health parameters with CNN-based ECG analysis.")
H("II. Problem Statement")
P("Manual ECG interpretation is time-consuming and depends on expert availability, while patient health "
  "parameters and ECG signals are usually analysed separately. The problem addressed is to provide an "
  "integrated, AI-assisted assessment that considers both sources of information for research and "
  "educational purposes.")
H("III. Objectives")
for o in ["Design an assessment system that uses patient health information and ECG signals.",
          "Classify ECG recordings into NORM, MI, STTC, HYP and CD using a CNN.",
          "Evaluate the model using accuracy, precision, recall, F1 score, ROC-AUC and confusion matrices.",
          "Identify strengths and weaknesses of the model per class."]:
    doc.add_paragraph(o, style="List Bullet")
H("IV. Existing System")
P(TODO % "describe existing approaches/related work with verified citations")
H("V. Proposed System")
P("The proposed system combines patient-data-based prediction with CNN-based ECG signal analysis to "
  "produce a heart disease risk assessment. " + TODO % "exact fusion/decision logic from source code")
H("VI. System Architecture")
P(TODO % "architecture description and diagram from project (UI, backend, model, patient-data module)")
H("VII. Methodology")
P(TODO % "end-to-end methodology as implemented")
H("VIII. Dataset Description")
P("The five classes NORM, MI, STTC, HYP and CD correspond to diagnostic superclasses. "
  + TODO % "dataset name, source, number of records, sampling rate, leads, split sizes")
TBL("Table 1 - Class information", "Class information (positive/negative counts derived from confusion matrices on the test set)")
H("IX. Data Preprocessing")
P(TODO % "actual preprocessing steps in code (label encoding, normalisation, splitting)")
H("X. ECG Signal Processing")
P(TODO % "actual signal processing steps (filtering, segmentation, scaling) used in code")
H("XI. CNN Model Architecture")
P("The ECG classifier is a CNN-based deep learning model. Layer details are listed in Table II.")
TBL("Table 2 - Model architecture", "CNN model architecture")
H("XII. Patient Health Parameter Processing")
P(TODO % "input parameters, encoding, and prediction method used for patient data")
H("XIII. Risk Assessment Workflow")
P(TODO % "step-by-step workflow from user input to displayed risk result")
H("XIV. Experimental Setup")
P(TODO % "hardware, Python/TensorFlow versions, split ratio")
H("XV. Model Training")
P(TODO % "optimizer, loss, batch size, epochs, callbacks from training script")
H("XVI. Evaluation Metrics")
P("Accuracy, precision, recall, F1 score, ROC-AUC and per-class confusion matrices (TN, FP, FN, TP) "
  "were used. Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R).")
H("XVII. Results and Discussion")
P("The model achieved a training accuracy of 91.07%, a validation accuracy of 88.12% and a test binary "
  "accuracy of 87.58%, with a mean ROC-AUC of 0.9052. The 3.49-point gap between training and test "
  "accuracy and the 0.54-point gap between validation and test accuracy indicate moderate generalisation "
  "with a small degree of overfitting. The test accuracy is below 90% and no claim of 90% test accuracy "
  "is made.")
TBL("Table 3 - Overall metrics", "Overall evaluation metrics")
FIG_(7, "Overall model performance.")
P("ROC-AUC exceeded 0.91 for NORM (0.9382), STTC (0.9305), CD (0.9178) and MI (0.9168), indicating good "
  "ranking ability. HYP obtained the lowest ROC-AUC (0.8225).")
H("XVIII. Graphical Analysis")
FIG_(1, "Training accuracy versus validation accuracy.")
FIG_(2, "Training loss versus validation loss.")
FIG_(3, "Class-wise precision, recall and F1 score.")
FIG_(4, "Class-wise ROC-AUC.")
H("XIX. Confusion Matrix Analysis")
TBL("Table 5 - Confusion matrix values", "Per-class confusion matrix values")
FIG_(5, "Multilabel confusion matrix for each ECG class.", 6.0)
H("XX. Class-wise Performance Analysis")
TBL("Table 4 - Class-wise metrics", "Class-wise accuracy, precision, recall, F1 score and ROC-AUC")
P("NORM had the highest recall (0.9138) and F1 (0.8503) but also the most false positives (205). "
  "STTC was balanced (precision 0.7679, recall 0.7629). MI and CD showed higher precision than recall. "
  "HYP had the lowest recall (0.2875) and F1 (0.3932): only 69 of 240 HYP cases were detected, making "
  "it the main target for future improvement.")
TBL("Table 6 - Error analysis", "Error analysis per class")
FIG_(6, "False positive versus false negative comparison per class.")
P("In ECG classification a false positive flags a healthy or unaffected recording as abnormal, which may "
  "cause unnecessary follow-up; a false negative misses an existing abnormality, which is generally the "
  "more serious error in screening. HYP (171 FN), MI (184 FN) and CD (147 FN) show notable false negatives.")
H("XXI. Limitations")
P("The system is an AI-assisted research/educational tool and not a clinical diagnostic device. "
  "Performance on HYP is limited, results are from a single dataset and split, and no external or "
  "clinical validation was performed. " + TODO % "any additional limitations from the project")
H("XXII. Future Scope")
P("Future work may address HYP performance (e.g. class-imbalance handling), external validation, "
  "threshold tuning per class and improved integration of patient parameters.")
H("XXIII. Conclusion")
P("The current system achieved 87.58% test binary accuracy and a mean ROC-AUC of 0.9052 for five ECG "
  "classes. Results are promising for an educational/research assessment tool, while HYP detection "
  "requires improvement. The system does not replace professional medical diagnosis.")
H("References")
P(TODO % "add only references you have verified (dataset paper, CNN/ECG papers, library docs)")

doc.save(os.path.join(OUT, "Heart_Disease_Risk_Assessment_Paper.docx"))

# ----------------------------------------------------------------------------
# 7. Provenance report
# ----------------------------------------------------------------------------
src = {
    "Fig 1 / Fig 2": hist_file or "NOT GENERATED - no history file found",
    "Fig 3, 4, 5, 6, 7 / Tables 3, 4, 5, 6": "Values supplied by author (class-wise metrics, confusion matrices, overall metrics)",
    "Table 1": "Class names + positive/negative counts derived from supplied confusion matrices",
    "Table 2": "Saved Keras model layers" if arch_rows else "PLACEHOLDER - fill from model.summary()",
}
with open(os.path.join(OUT, "data_sources.txt"), "w") as f:
    for k, v in src.items():
        f.write(f"{k}: {v}\n")
print("Done. See", OUT)
