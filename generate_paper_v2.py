"""
generate_paper_v2.py - Heart Disease Risk Assessment Agent
Run from the PROJECT ROOT (the folder that contains results/ and your .py files):

    pip install matplotlib numpy pandas python-docx
    python generate_paper_v2.py

What it does
  1. Scans your project source (*.py) and extracts the REAL CNN layers, optimizer, learning
     rate, loss, epochs, batch size and callbacks -> paper_output/extracted_project_details.json
     Anything it cannot find is written into the paper as [CONFIRM FROM PROJECT: ...].
  2. Uses your existing graphs in results/ (improved_accuracy_curve.png, improved_loss_curve.png,
     multilabel_confusion_matrix.png, classwise_performance.png, classwise_roc_auc.png,
     classwise_errors.png). Only if one is missing does it redraw it from the supplied
     actual values (never the training curves - those are never redrawn or invented).
  3. Draws the system-architecture and workflow diagrams.
  4. Opens your reference paper ("Research paper_updated*.docx") ONLY for its page layout and
     styles, removes its content, and writes the heart-disease paper into it.
     If the file is not found, a plain default Word template is used.
  5. Writes paper_output/data_sources.txt (which file/values produced each figure and table).
"""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT, FIG = "paper_output", "paper_output/figures"
os.makedirs(FIG, exist_ok=True)
RES = "results"
SKIP = ("venv", ".venv", "site-packages", "node_modules", OUT, "generate_paper", "dataset")

# ---------------------------------------------------------------- ACTUAL RESULTS
CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]
OVERALL = {"Training Accuracy": 91.07, "Validation Accuracy": 88.12,
           "Test Binary Accuracy": 87.58, "Mean ROC-AUC": 90.52}
CW = {"NORM": (85.29, .7950, .9138, .8503, .9382), "MI": (85.81, .7707, .6110, .6816, .9168),
      "STTC": (88.60, .7679, .7629, .7654, .9305), "HYP": (88.81, .6216, .2875, .3932, .8225),
      "CD": (89.39, .8333, .6517, .7314, .9178)}
CM = {"NORM": (828, 205, 75, 795), "MI": (1344, 86, 184, 289), "STTC": (1332, 107, 110, 354),
      "HYP": (1621, 42, 171, 69), "CD": (1426, 55, 147, 275)}
N_TEST = {sum(v) for v in CM.values()}          # all classes must share the same test size
assert len(N_TEST) == 1, "confusion matrices have different totals"
N_TEST = N_TEST.pop()
POS = {c: CM[c][2] + CM[c][3] for c in CLASSES}  # actual positives per class in test set

# figure numbers (results figures fixed as you requested; diagrams appear earlier in the text)
FN = dict(acc=1, loss=2, prf=3, cm=4, roc=5, err=6, overall=7, arch=8, flow=9)

# ---------------------------------------------------------------- SOURCE SCAN
def py_files():
    return [f for f in glob.glob("**/*.py", recursive=True) if not any(s in f for s in SKIP)]

def call_args(src, i):            # i = index of '(' ; returns text up to matching ')'
    d = 0
    for j in range(i, len(src)):
        d += (src[j] == "(") - (src[j] == ")")
        if d == 0:
            return src[i + 1:j]
    return ""

LAYER = re.compile(r"\b(Conv1D|Conv2D|SeparableConv1D|MaxPooling1D|MaxPool1D|AveragePooling1D|"
                   r"GlobalAveragePooling1D|GlobalMaxPooling1D|BatchNormalization|Dropout|"
                   r"SpatialDropout1D|Dense|Flatten|LSTM|GRU|Bidirectional|Activation|ReLU|LeakyReLU)\s*\(")
PATS = {"optimizer": r"optimizer\s*=\s*([\w\.\"']+(?:\([^)]*\))?)",
        "loss": r"\bloss\s*=\s*([\w\.\"']+(?:\([^)]*\))?)",
        "learning_rate": r"(?:learning_rate|\blr)\s*=\s*([0-9.eE\-]+)",
        "epochs": r"\bepochs\s*=\s*(\d+)", "batch_size": r"\bbatch_size\s*=\s*(\d+)"}
CALLBACKS = r"\b(EarlyStopping|ReduceLROnPlateau|ModelCheckpoint|LearningRateScheduler)\s*\("

def scan_project():
    info = {"layers": [], "layers_file": None, "callbacks": [], "web_framework": None,
            "mentions_ptbxl": False, "files_scanned": []}
    best = 0
    for f in py_files():
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        info["files_scanned"].append(f)
        ls = [(m.group(1), " ".join(call_args(s, m.end() - 1).split())[:90]) for m in LAYER.finditer(s)]
        if len(ls) > best:
            best, info["layers"], info["layers_file"] = len(ls), ls, f
        for k, p in PATS.items():
            for m in re.finditer(p, s):
                info.setdefault(k, {}).setdefault(m.group(1).strip(), f)
        for m in re.finditer(CALLBACKS, s):
            if m.group(1) not in info["callbacks"]:
                info["callbacks"].append(m.group(1))
        low = s.lower()
        for fw in ("streamlit", "flask", "fastapi", "gradio", "django"):
            if info["web_framework"] is None and re.search(r"(import|from)\s+" + fw, low):
                info["web_framework"] = fw.capitalize()
        if "ptb" in low or "ptbxl" in low:
            info["mentions_ptbxl"] = True
    return info

INFO = scan_project()
json.dump(INFO, open(f"{OUT}/extracted_project_details.json", "w"), indent=2)

def got(key, label=None):
    """value found in source, or a visible placeholder"""
    v = INFO.get(key)
    if v:
        return ", ".join(v.keys())
    return f"[CONFIRM FROM PROJECT: {label or key}]"

# ---------------------------------------------------------------- MATPLOTLIB STYLE
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
G = ["#222222", "#666666", "#aaaaaa"]
X = np.arange(len(CLASSES))
SRC = {}                       # provenance

def save(fig, name):
    p = f"{FIG}/{name}"; fig.tight_layout(); fig.savefig(p, dpi=300); plt.close(fig); return p

def existing(fname):
    p = os.path.join(RES, fname)
    return p if os.path.exists(p) else None

def use_or_make(key, fname, maker):
    p = existing(fname)
    if p:
        SRC[key] = f"existing project file {p}"; return p
    if maker is None:
        SRC[key] = f"NOT AVAILABLE - {RES}/{fname} not found (not redrawn, to avoid invented data)"; return None
    SRC[key] = f"redrawn from supplied actual values ({RES}/{fname} not found)"; return maker()

def mk_prf():
    fig, ax = plt.subplots(figsize=(3.5, 2.8)); w = .26
    for i, (l, k) in enumerate([("Precision", 1), ("Recall", 2), ("F1-Score", 3)]):
        v = [CW[c][k] for c in CLASSES]
        b = ax.bar(X + (i - 1) * w, v, w, label=l, color=G[i], edgecolor="black", lw=.5)
        for r, y in zip(b, v): ax.text(r.get_x() + w / 2, y + .01, f"{y:.2f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(X); ax.set_xticklabels(CLASSES); ax.set(xlabel="ECG class", ylabel="Score", ylim=(0, 1.15))
    ax.legend(ncol=3, loc="upper center"); return save(fig, "classwise_performance.png")

def mk_roc():
    fig, ax = plt.subplots(figsize=(3.5, 2.6)); v = [CW[c][4] for c in CLASSES]
    b = ax.bar(CLASSES, v, color=G[1], edgecolor="black", lw=.5)
    for r, y in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, y + .005, f"{y:.4f}", ha="center", fontsize=8)
    ax.set(xlabel="ECG class", ylabel="ROC-AUC", ylim=(.5, 1)); return save(fig, "classwise_roc_auc.png")

def mk_cm():
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
    for ax, c in zip(axes.flat, CLASSES):
        tn, fp, fn, tp = CM[c]; m = np.array([[tn, fp], [fn, tp]])
        ax.imshow(m, cmap="Greys", vmin=0, vmax=m.max() * 1.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=11, color="white" if m[i, j] > m.max() * .6 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=c); ax.grid(False)
    axes.flat[-1].axis("off"); return save(fig, "multilabel_confusion_matrix.png")

def mk_err():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    b1 = ax.bar(X - .2, [CM[c][1] for c in CLASSES], .4, label="False Positives", color=G[1], edgecolor="black", lw=.5)
    b2 = ax.bar(X + .2, [CM[c][2] for c in CLASSES], .4, label="False Negatives", color=G[2], edgecolor="black", lw=.5, hatch="//")
    for bs in (b1, b2):
        for r in bs: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 3, int(r.get_height()), ha="center", fontsize=7)
    ax.set_xticks(X); ax.set_xticklabels(CLASSES); ax.set(xlabel="ECG class", ylabel="Samples"); ax.legend()
    return save(fig, "classwise_errors.png")

def mk_overall():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    lab = ["Train\nAcc.", "Val.\nAcc.", "Test\nAcc.", "Mean\nROC-AUC"]
    v = [91.07, 88.12, 87.58, 90.52]
    b = ax.bar(lab, v, color=G[1], edgecolor="black", lw=.5)
    for r, y in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, y + .3, f"{y:.2f}%", ha="center", fontsize=8)
    ax.set(ylabel="Percentage (%)", ylim=(80, 95)); return save(fig, "overall_performance.png")

def flow_diagram(title, steps, fname, two_input=False):
    n = len(steps); fig, ax = plt.subplots(figsize=(3.4, 0.62 * n + .3)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, n)
    for i, s in enumerate(steps):
        y = n - i - 1
        ax.add_patch(FancyBboxPatch((.08, y + .18), .84, .62, boxstyle="round,pad=0.02", fc="white", ec="black", lw=.8))
        ax.text(.5, y + .49, s, ha="center", va="center", fontsize=7.5)
        if i < n - 1: ax.annotate("", xy=(.5, y + .16), xytext=(.5, y - .02), arrowprops=dict(arrowstyle="->", lw=.8))
    return save(fig, fname)

def mk_arch():
    ui = f"Web Interface ({INFO['web_framework']})" if INFO["web_framework"] else "Web Interface"
    return flow_diagram("", ["Patient Health Information  +  ECG Signal / Input", "Data Preprocessing",
                             "Patient Data Processing  +  ECG Signal Processing", "CNN-based ECG Classification",
                             "Risk Assessment", "Prediction / Risk Percentage",
                             "Health Awareness / Recommendations", ui], "system_architecture.png")

def mk_flow():
    return flow_diagram("", ["Input", "Data Collection", "Preprocessing", "Feature / Signal Processing",
                             "CNN Model", "Prediction", "Risk Assessment", "Result and Health Awareness"], "workflow.png")

figs = {
    "acc": use_or_make("Fig. 1", "improved_accuracy_curve.png", None),
    "loss": use_or_make("Fig. 2", "improved_loss_curve.png", None),
    "prf": use_or_make("Fig. 3", "classwise_performance.png", mk_prf),
    "cm": use_or_make("Fig. 4", "multilabel_confusion_matrix.png", mk_cm),
    "roc": use_or_make("Fig. 5", "classwise_roc_auc.png", mk_roc),
    "err": use_or_make("Fig. 6", "classwise_errors.png", mk_err),
    "overall": mk_overall(), "arch": mk_arch(), "flow": mk_flow()}
SRC["Fig. 7"] = "generated from the four overall values supplied (91.07, 88.12, 87.58, 0.9052)"
SRC["Fig. 8 / Fig. 9"] = "generated diagrams; component names taken from the flow you specified" + \
    (f"; web framework detected in code: {INFO['web_framework']}" if INFO["web_framework"] else "; web framework not detected")

# ---------------------------------------------------------------- WORD DOCUMENT
def new_doc():
    ref = [f for f in glob.glob("*.docx") + glob.glob("**/*.docx", recursive=True)
           if "research paper" in f.lower() and OUT not in f]
    if ref:
        d = Document(ref[0]); body = d.element.body
        for el in list(body):
            if not el.tag.endswith("sectPr"): body.remove(el)
        SRC["Template"] = f"styles/page layout of {ref[0]} (content removed)"
        return d
    SRC["Template"] = "default Word template (reference paper not found in project folder)"
    return Document()

doc = new_doc()
try: doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(10)
except Exception: pass

def H(t, lvl=1):
    try: doc.add_paragraph(t, style=f"Heading {lvl}")
    except KeyError: doc.add_paragraph().add_run(t).bold = True
def P(t):
    p = doc.add_paragraph(t); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; return p
def CAP(t):
    try: p = doc.add_paragraph(t, style="Caption")
    except KeyError: p = doc.add_paragraph(t)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
def FIGURE(key, cap, w=3.3):
    if figs.get(key):
        doc.add_picture(figs[key], width=Inches(w)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        CAP(f"Fig. {FN[key]}. {cap}")
    else:
        P(f"[Fig. {FN[key]} not inserted: results/ file for '{cap}' was not found. Place the project graph in results/ and re-run.]")
def TABLE(num, cap, head, rows):
    CAP(f"TABLE {num}. {cap}")
    try: t = doc.add_table(rows=1, cols=len(head), style="Table Grid")
    except KeyError: t = doc.add_table(rows=1, cols=len(head))
    for i, h in enumerate(head):
        t.rows[0].cells[i].text = str(h)
        for r in t.rows[0].cells[i].paragraphs[0].runs: r.bold = True
    for row in rows:
        cs = t.add_row().cells
        for i, v in enumerate(row): cs[i].text = str(v)
    doc.add_paragraph()

CONF = lambda x: f"[CONFIRM FROM PROJECT: {x}]"
# ---- references (real, verify details before submission) ----
REFS = [
 'Y. LeCun, Y. Bengio and G. Hinton, "Deep learning," Nature, vol. 521, pp. 436-444, 2015.',
 'P. Rajpurkar, A. Y. Hannun, M. Haghpanahi, C. Bourn and A. Y. Ng, "Cardiologist-level arrhythmia detection with convolutional neural networks," arXiv:1707.01836, 2017.',
 'A. Y. Hannun et al., "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network," Nature Medicine, vol. 25, no. 1, pp. 65-69, 2019.',
 'A. H. Ribeiro et al., "Automatic diagnosis of the 12-lead ECG using a deep neural network," Nature Communications, vol. 11, Art. 1760, 2020.',
 'S. Kiranyaz, T. Ince and M. Gabbouj, "Real-time patient-specific ECG classification by 1-D convolutional neural networks," IEEE Trans. Biomed. Eng., vol. 63, no. 3, pp. 664-675, 2016.',
 'P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," Scientific Data, vol. 7, Art. 154, 2020.',
 'N. Strodthoff, P. Wagner, T. Schaeffter and W. Samek, "Deep learning for ECG analysis: Benchmarks and insights from PTB-XL," IEEE J. Biomed. Health Inform., vol. 25, no. 5, pp. 1519-1528, 2021.',
 'T. Fawcett, "An introduction to ROC analysis," Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.',
 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, e0118432, 2015.']

doc.add_paragraph().add_run("Heart Disease Risk Assessment Agent: An AI-Assisted System Using Patient Health "
                            "Information and CNN-Based ECG Classification").bold = True
H("Abstract", 1)
P("Cardiovascular disease remains a major cause of mortality, and early awareness of cardiac risk is important. "
  "This paper presents the Heart Disease Risk Assessment Agent, an AI-assisted system for preliminary and educational "
  "risk assessment that combines patient health information with electrocardiogram (ECG) analysis through a "
  "convolutional neural network (CNN) and a web-based interface. The ECG model performs multilabel classification of "
  "five classes: NORM, MI, STTC, HYP and CD. The current implementation reached a training accuracy of 91.07%, a "
  "validation accuracy of 88.12%, a test binary accuracy of 87.58% and a mean ROC-AUC of 0.9052. Class-wise ROC-AUC "
  "ranged from 0.8225 (HYP) to 0.9382 (NORM), while HYP showed the lowest recall (0.2875) and F1-score (0.3932). "
  "The system is not clinically validated and is not a replacement for professional medical diagnosis.")
H("Keywords", 1)
P("Heart disease risk assessment; ECG classification; convolutional neural network; deep learning; multilabel "
  "classification; ROC-AUC; AI-assisted healthcare.")

H("1. Introduction", 1)
P("Cardiovascular diseases are among the leading causes of death worldwide, and the 12-lead ECG is a widely used, "
  "non-invasive tool for examining cardiac electrical activity. Interpreting ECGs manually is time-consuming and "
  "depends on specialist availability, which motivates computer-assisted analysis. Deep learning, and CNNs in "
  "particular, can learn discriminative patterns directly from signals [1], [2].")
P("This work presents the Heart Disease Risk Assessment Agent, which integrates patient health information with "
  "CNN-based ECG classification behind a web interface. The system is intended for preliminary, educational "
  "assessment only and does not provide a clinical diagnosis. The main contributions are: (i) an integrated "
  "assessment pipeline combining patient data and ECG analysis; (ii) a CNN-based multilabel ECG classifier for five "
  "classes; and (iii) a per-class evaluation using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.")

H("2. Literature Survey", 1)
P("Deep learning has been applied extensively to ECG analysis. Rajpurkar et al. proposed a deep convolutional network "
  "for arrhythmia detection from single-lead ECG [2], and Hannun et al. reported cardiologist-level rhythm "
  "classification on ambulatory recordings [3]. Ribeiro et al. extended this to automatic diagnosis of abnormalities "
  "in 12-lead ECGs [4]. Kiranyaz et al. showed that one-dimensional CNNs can operate directly on ECG waveforms [5].")
P("For diagnostic-class benchmarking, the PTB-XL dataset provides a large public collection of 12-lead ECGs annotated "
  "with diagnostic superclasses [6], and Strodthoff et al. benchmarked several deep architectures on it [7]. "
  "Evaluation of such classifiers commonly relies on ROC analysis [8]; on imbalanced data, precision and recall "
  "provide complementary information [9]. The present work follows this general direction but focuses on an "
  "integrated, web-based preliminary risk-assessment system that also considers patient health information.")

H("3. Proposed Methodology", 1)
H("3.1 Dataset", 2)
ds = ("The ECG class labels used are NORM (normal ECG), MI (myocardial infarction), STTC (ST/T change), HYP "
      "(hypertrophy) and CD (conduction disturbance), which correspond to the diagnostic superclasses defined for "
      "PTB-XL [6]. " if INFO["mentions_ptbxl"] else
      "The ECG class labels used are NORM, MI, STTC, HYP and CD. " + CONF("dataset name and source") + " ")
P(ds + f"The confusion matrices reported in this paper were computed on a test set of {N_TEST} records. "
  f"Dataset size, training/validation split sizes, sampling rate and leads used: {CONF('from data-loading code')}.")
TABLE(1, "Overall Model Performance", ["Metric", "Value"],
      [["Training Accuracy", "91.07%"], ["Validation Accuracy", "88.12%"],
       ["Test Binary Accuracy", "87.58%"], ["Mean ROC-AUC", "0.9052"]])
P("Table 1 summarises the overall performance. The number of positive test records per class, derived from the "
  "confusion matrices, is: " + ", ".join(f"{c} = {POS[c]}" for c in CLASSES) +
  f" (out of {N_TEST}); HYP is the least represented class in the test set ({POS['HYP']/N_TEST*100:.1f}%).")
H("3.2 Data Preprocessing", 2)
P(CONF("actual preprocessing steps: label encoding, normalisation, splitting, etc."))
H("3.3 ECG Signal Processing", 2)
P(CONF("actual signal processing steps: filtering, segmentation, scaling, etc."))
H("3.4 Patient Health Parameter Processing", 2)
P(CONF("patient input parameters, encoding, and how they are used in the risk assessment"))
H("3.5 CNN Model Architecture", 2)
if INFO["layers"]:
    P(f"The CNN layers identified in the project source ({INFO['layers_file']}) are listed in Table 2 in the order "
      "they appear in the code. Layer arguments are shown as written in the source.")
    TABLE(2, "CNN Model Layers (from project source)", ["#", "Layer", "Arguments (as in code)"],
          [[i + 1, t, a] for i, (t, a) in enumerate(INFO["layers"])])
else:
    P(CONF("CNN layers - paste model.summary() or the model-building code; none could be detected automatically"))
H("3.6 Training Configuration", 2)
P(f"Optimizer: {got('optimizer')}. Learning rate: {got('learning_rate')}. Loss function: {got('loss')}. "
  f"Epochs: {got('epochs')}. Batch size: {got('batch_size')}. Callbacks: "
  f"{', '.join(INFO['callbacks']) if INFO['callbacks'] else CONF('callbacks')}.")

H("4. System Architecture and Implementation", 1)
P("Fig. 8 shows the overall architecture: patient health information and the ECG input are preprocessed and "
  "processed separately, the CNN performs ECG classification, and the risk assessment result is presented together "
  "with health-awareness information through the web interface.")
FIGURE("arch", "System architecture of the Heart Disease Risk Assessment Agent.")
P("Fig. 9 shows the workflow from input to the final result.")
FIGURE("flow", "Workflow of the risk assessment process.")
P(f"Implementation details (language, libraries, web framework"
  + (f"; detected: {INFO['web_framework']}" if INFO["web_framework"] else "") + f"): {CONF('confirm')}.")

H("5. Results and Discussion", 1)
H("5.1 Model Performance", 2)
P("The model reached a training accuracy of 91.07%, a validation accuracy of 88.12% and a test binary accuracy of "
  "87.58%, with a mean ROC-AUC of 0.9052 (Table 1, Fig. 7). Training accuracy exceeds validation accuracy by 2.95 "
  "percentage points and test accuracy by 3.49 points, while validation and test accuracy differ by only 0.54 "
  "points. This indicates a modest generalisation gap; the test accuracy is below 90% and no higher figure is claimed.")
FIGURE("overall", "Overall model performance (training, validation and test accuracy, mean ROC-AUC).")
H("5.2 Training and Validation Performance", 2)
P("Fig. 1 shows the accuracy of the model on the training and validation data across epochs, and thus its learning "
  "behaviour. Fig. 2 shows the training and validation loss, which reflects the prediction error during training. "
  "Both figures are taken directly from the project's training history.")
FIGURE("acc", "Training accuracy versus validation accuracy.")
FIGURE("loss", "Training loss versus validation loss.")
H("5.3 Confusion Matrix", 2)
P("The confusion matrix analyses correct and incorrect predictions for each ECG class. True positives (TP) are "
  "abnormal or labelled records correctly identified, true negatives (TN) are correctly rejected records, false "
  "positives (FP) are records flagged for a class they do not have, and false negatives (FN) are records with the "
  "class that the model missed. Values are given in Table 3 and Fig. 4.")
TABLE(3, "Confusion Matrix Values", ["Class", "TN", "FP", "FN", "TP"], [[c, *CM[c]] for c in CLASSES])
FIGURE("cm", "Class-wise (multilabel) confusion matrices.", 3.4)
fnr = {c: CM[c][2] / POS[c] * 100 for c in CLASSES}
P(f"MI has the largest number of false negatives ({CM['MI'][2]}), followed by HYP ({CM['HYP'][2]}) and CD "
  f"({CM['CD'][2]}). Relative to the number of positive records, HYP misses the largest share "
  f"({fnr['HYP']:.1f}%: {CM['HYP'][2]} of {POS['HYP']}), whereas NORM misses the smallest ({fnr['NORM']:.1f}%). "
  f"NORM has the most false positives ({CM['NORM'][1]}). In ECG classification, a false negative means an existing "
  "abnormality goes undetected, which is generally the more serious error in a screening context, whereas a false "
  "positive may lead to unnecessary follow-up. Reducing false negatives is therefore an important goal.")
TABLE(4, "Error Analysis", ["Class", "False Positive", "False Negative"], [[c, CM[c][1], CM[c][2]] for c in CLASSES])
FIGURE("err", "False positive versus false negative comparison for each class.")
H("5.4 Classification Results", 2)
TABLE(5, "Class-wise Performance", ["Class", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
      [[c, f"{v[0]:.2f}%", f"{v[1]:.4f}", f"{v[2]:.4f}", f"{v[3]:.4f}", f"{v[4]:.4f}"] for c, v in CW.items()])
FIGURE("prf", "Precision, recall and F1-score for NORM, MI, STTC, HYP and CD.")
FIGURE("roc", "Class-wise ROC-AUC.")
P("NORM achieved the strongest recall (0.9138) and F1-score (0.8503) among the classes, with the highest ROC-AUC "
  "(0.9382). STTC showed balanced precision (0.7679) and recall (0.7629). CD achieved the highest precision (0.8333) "
  "but a lower recall (0.6517). MI had a precision of 0.7707 and a recall of 0.6110. HYP showed the weakest "
  "recall (0.2875) and F1-score (0.3932) and the lowest ROC-AUC (0.8225): only 69 of 240 HYP records were detected.")
H("5.5 Discussion", 2)
P("ROC-AUC values above 0.91 for four of the five classes indicate good ranking ability, whereas the HYP result "
  "shows that threshold-dependent measures (recall, F1) can be low even when ranking ability is moderate. Two "
  "factors may contribute to the weaker HYP and MI results. First, class imbalance: HYP is the least represented "
  f"class in the test set ({POS['HYP']} of {N_TEST} records), so it contributes fewer examples for the model to learn "
  "from and its errors are penalised less in overall accuracy. Second, similarity between ECG patterns: "
  "morphological changes associated with hypertrophy, ST/T changes and conduction abnormalities can overlap, which "
  "may make some classes harder to separate. These are plausible explanations and were not tested separately in this "
  "work. Per-class accuracies (85.29%-89.39%) should be read together with recall and precision, since accuracy "
  "alone can be high for a rarely positive class. All results come from a single dataset split; the system has not "
  "been clinically validated.")

H("6. Conclusion and Future Scope", 1)
P("This paper presented the Heart Disease Risk Assessment Agent, an AI-assisted system that combines patient health "
  "information with CNN-based ECG classification and a web interface. The current implementation achieved a test "
  "binary accuracy of 87.58% and a mean ROC-AUC of 0.9052, which supports its use for preliminary, educational "
  "assessment. Limitations include weak detection of HYP (recall 0.2875) and MI (recall 0.6110), evaluation on a "
  "single dataset split, and the absence of clinical or external validation. The system must not be used as a "
  "replacement for professional medical diagnosis.")
P("Future work (not current features) may include: larger and more diverse ECG datasets; improved handling of class "
  "imbalance; better detection of HYP and MI cases; improved generalisation; external validation; explainable AI "
  "methods; and deployment improvements.")

H("References", 1)
for i, r in enumerate(REFS, 1):
    doc.add_paragraph(f"[{i}] {r}")

doc.save(f"{OUT}/Heart_Disease_Risk_Assessment_Paper_v2.docx")

# ---------------------------------------------------------------- PROVENANCE
SRC["Table 1, 3, 4, 5"] = "actual values supplied (overall metrics, confusion matrices, class-wise metrics)"
SRC["Table 2 / Section 3.6"] = (f"parsed from {INFO['layers_file']} and other .py files" if INFO["layers"]
                                else "NOT FOUND in source - placeholder")
SRC["Test set size / positives per class"] = "derived arithmetically from supplied confusion matrices"
SRC["Files scanned"] = f"{len(INFO['files_scanned'])} Python files (see extracted_project_details.json)"
with open(f"{OUT}/data_sources.txt", "w") as f:
    for k, v in SRC.items(): f.write(f"{k}: {v}\n")
print("Done ->", OUT); print(open(f"{OUT}/data_sources.txt").read())
print("Search the .docx for '[CONFIRM FROM PROJECT' to find everything still to fill in.")
