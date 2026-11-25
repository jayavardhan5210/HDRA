"""
make_architecture_workflow_style.py
Colorful workflow-style architecture diagram (cylinder / parallelograms / diamond / colored blocks),
in the visual style of the reference paper's "Proposed Workflow" figure.
All components come from the verified source code and dataset statistics.

    pip install matplotlib
    python make_architecture_workflow_style.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Polygon, Ellipse, FancyBboxPatch, FancyArrow

# Select high readability font
names = {f.name for f in fm.fontManager.ttflist}
FONT = "DejaVu Sans"
for cand in ("Arial Narrow", "Liberation Sans Narrow", "DejaVu Sans", "Arial"):
    if cand in names:
        FONT = cand
        break
plt.rcParams["font.family"] = FONT

fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=100)
ax.set_xlim(0, 100)
ax.set_ylim(-2, 60)
ax.axis("off")
fig.patch.set_facecolor("white")

FS = 6.0
EC = "#222222"

def txt(x, y, s, color="black", fs=FS, style="normal", weight="bold"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, fontweight=weight, color=color,
            style=style, linespacing=1.2, zorder=5)

def rect(x0, x1, y0, y1, s, fc, color="black", fs=FS):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=EC, lw=0.9, zorder=2))
    txt((x0 + x1) / 2, (y0 + y1) / 2, s, color, fs=fs)

def para(x0, x1, y0, y1, s, fc, skew=2.0, fs=FS):
    ax.add_patch(Polygon([(x0 + skew, y1), (x1, y1), (x1 - skew, y0), (x0, y0)], fc=fc, ec=EC, lw=0.9, zorder=2))
    txt((x0 + x1) / 2, (y0 + y1) / 2, s, fs=fs)

def diamond(cx, cy, w, h, s, fc, color="white", fs=FS):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], fc=fc, ec=EC, lw=0.9, zorder=2))
    txt(cx, cy, s, color, fs=fs)

def cylinder(x0, x1, y0, y1, s, fc, fs=FS):
    e = 2.2
    w = x1 - x0
    ax.add_patch(Rectangle((x0, y0 + e / 2), w, (y1 - y0) - e, fc=fc, ec="none", zorder=2))
    ax.add_patch(Ellipse(((x0 + x1) / 2, y0 + e / 2), w, e, fc=fc, ec=EC, lw=0.9, zorder=2))
    ax.plot([x0, x0], [y0 + e / 2, y1 - e / 2], color=EC, lw=0.9, zorder=3)
    ax.plot([x1, x1], [y0 + e / 2, y1 - e / 2], color=EC, lw=0.9, zorder=3)
    ax.add_patch(Ellipse(((x0 + x1) / 2, y1 - e / 2), w, e, fc=fc, ec=EC, lw=0.9, zorder=4))
    txt((x0 + x1) / 2, (y0 + y1) / 2 - 0.5, s, fs=fs)

def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", shrinkA=0, shrinkB=0,
                                mutation_scale=7, linestyle=ls))

def elbow(pts):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.9, zorder=1)
    arrow(*pts[-2], *pts[-1])

# ---------------- Section Labels & Dividing Rules
ax.text(11, 58.3, "Risk assessment pipeline (inference)", fontsize=6.2, style="italic", color="#444444", va="center", fontweight="bold")
ax.plot([11, 100], [55.6, 55.6], color="#BBBBBB", lw=0.5, ls="--")
ax.text(11, 19.6, "Model development (PTB-XL training and evaluation)", fontsize=6.2, style="italic", color="#444444", va="center", fontweight="bold")
ax.plot([11, 100], [21.4, 21.4], color="#BBBBBB", lw=0.5, ls="--")

# Row Coordinates
YC = (41, 53)  # clinical row
YE = (26, 38)  # ECG row
mc, me = sum(YC) / 2, sum(YE) / 2

# ---------------- Column 0: User Interface
rect(0, 9, YE[0], YC[1], "User /\nWeb\nInterface\n(Flask)", "#FF9EC7", fs=6.0)

# ---------------- Column 1: Multi-Modal Inputs
para(11.5, 22.5, YC[0], YC[1], "Patient\nHealth Data\n(8 parameters)", "#3D9BFF", fs=5.8)
para(11.5, 22.5, YE[0], YE[1], "12-Lead ECG\nPrintout Image\n(JPG / PNG)", "#3D9BFF", fs=5.8)
elbow([(9, mc), (11.5 + 1.2, mc)])
elbow([(9, me), (11.5 + 1.2, me)])

# ---------------- Column 2: Preprocessing
rect(25.5, 36.5, YC[0], YC[1], "Clinical Data\nPreprocessing\n(normalize\nto [0, 1])", "#B6F44E", fs=5.7)
rect(25.5, 36.5, YE[0], YE[1], "ECG Image\nPreprocessing\n(OpenCV masking,\nmorph. filtering)", "#B6F44E", fs=5.6)
arrow(22.5 - 1.0, mc, 25.5, mc)
arrow(22.5 - 1.0, me, 25.5, me)

# ---------------- Column 3: Feature & Signal Processing
rect(39.5, 50.5, YC[0], YC[1], "Clinical Scoring\nEngine\nS_clinical =\nΣ w_k · s_k", "#FFE94D", fs=5.7)
rect(39.5, 50.5, YE[0], YE[1], "ECG Signal\nProcessing\n(1000 × 12\narray)", "#22F0F0", fs=5.8)
arrow(36.5, mc, 39.5, mc)
arrow(36.5, me, 39.5, me)

# ---------------- Column 4: CNN Inference & Non-ML Scoring Note
rect(53.5, 64.5, YE[0], YE[1], "Trained\n1D-CNN\n5 sigmoid outputs:\nNORM, MI, STTC,\nHYP, CD", "#A594FF", fs=5.4)
arrow(50.5, me, 53.5, me)
ax.text(59, mc, "Deterministic weighted\nscoring (no trained\nML model)", ha="center", va="center",
        fontsize=5.3, style="italic", color="#444444")

# ---------------- Column 5: Decision Fusion Agent
fy0, fy1 = YE[0] + 1, YC[1] - 1
rect(67.5, 78.5, fy0, fy1, "Heart Disease\nRisk Assessment\nAgent\n\nDecision Fusion\nS_combined =\n0.40 · S_clinical\n+ 0.60 · C_ECG\n(clip 0.05 – 0.98)", "#FF9A1F", fs=5.3)

# Routing arrows to Fusion Agent
elbow([(50.5, mc + 3.2), (65.5, mc + 3.2), (65.5, fy1 - 3.8), (67.5, fy1 - 3.8)])
arrow(64.5, me + 2, 67.5, me + 2)
ax.text(66.0, me + 3.6, "C_ECG", fontsize=5.2, ha="center", color="#333333", style="italic", fontweight="bold")
ax.text(58.0, mc + 4.4, "S_clinical", fontsize=5.2, ha="center", color="#333333", style="italic", fontweight="bold")

# ---------------- Column 6: Risk Tiers
tiers = [
    ("Low\n< 0.35", "#CCFF90", 47.5),
    ("Medium\n0.35 – 0.70", "#FFD54F", 39.5),
    ("High\n≥ 0.70", "#FF5A5A", 31.5)
]
for s, c, yc in tiers:
    rect(82, 91, yc - 3.2, yc + 3.2, s, c, fs=5.6)
    arrow(78.5, (fy0 + fy1) / 2, 82, yc)

# ---------------- Column 7: Multi-Component Results
rect(93.5, 100, fy0 + 0.2, fy1 - 0.2, "Results\n\nRisk tier &\nHealth tips\n\nRisk-factor\nchart\n(Matplotlib)\n\nJSON\nresponse", "#69F0AE", fs=5.1)
for _, _, yc in tiers:
    arrow(91, yc, 93.5, yc)

# =========================================================================
# MODEL DEVELOPMENT SECTION (BOTTOM ROW)
# =========================================================================
cylinder(11.5, 22.5, 2, 15, "PTB-XL\nDataset\n(21,837\nECGs)", "#FF3E9A", fs=5.7)

ax.add_patch(FancyArrow(22.7, 8.5, 3.0, 0, width=1.6, head_width=3.4, head_length=1.6, length_includes_head=True,
                        fc="#3DFFA0", ec=EC, lw=0.8, zorder=2))

diamond(32.5, 8.5, 10, 14, "Patient-\nwise\nSplit", "#6A00FF", color="white", fs=5.6)

rect(39.5, 50.5, 11, 18, "Training Data\n(Folds 1–8)\n14,234", "#B6F44E", fs=5.5)
rect(39.5, 50.5, -1, 6, "Validation (Fold 9)\n1,891\nTest (Fold 10)\n1,903", "#FFE94D", fs=5.4)

arrow(35.5, 12.5, 39.5, 14.5)
arrow(35.5, 4.5, 39.5, 2.5)

rect(53.5, 64.5, 10.5, 18.5, "CNN Training\nFocal loss, Adam\nη = 3e-4, batch 32", "#22F0F0", fs=5.5)
arrow(50.5, 14.5, 53.5, 14.5)

rect(53.5, 64.5, -1, 7, "Model Evaluation\nAccuracy, ROC-AUC,\nconfusion matrix", "#FF9A1F", fs=5.4)
arrow(50.5, 2.5, 53.5, 2.5)
arrow(59, 10.5, 59, 7)

# Trained weights connection arrow to inference 1D-CNN
elbow([(64.5, 14.5), (68, 14.5), (68, 22.5)])
ax.annotate("", xy=(64.5 - 2.0, me - 6.5), xytext=(68, 22.5), 
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", mutation_scale=7,
                            connectionstyle="angle,angleA=0,angleB=90,rad=0"), zorder=1)
ax.text(72, 12.5, "trained weights\nused for inference", fontsize=5.3, style="italic", color="#444444", va="center")

fig.savefig("system_architecture_workflow.png", dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig("system_architecture_workflow.pdf", bbox_inches="tight", facecolor="white")
print(f"saved system_architecture_workflow.png / .pdf (font: {FONT})")
