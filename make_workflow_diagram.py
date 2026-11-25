import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

def create_proposed_workflow_diagram():
    # Configure publication fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
    plt.rcParams["font.size"] = 9.0

    # Dimensions for single-column/two-column paper figure
    fig_w, fig_h = 10.5, 14.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    # Workflow Steps Data (Top to Bottom)
    steps = [
        {
            "step": "STEP 1: MULTI-MODAL DATA INPUT ACQUISITION",
            "title": "Patient Health Profile & 12-Lead ECG Printout",
            "details": [
                "• 8 Clinical Parameters: Age, Systolic BP, Cholesterol, BMI, Diabetes, Smoking, Family History, Exercise",
                "• 12-Lead Diagnostic ECG Image: Paper recording in standard 3×4 printed layout (JPG / JPEG / PNG)"
            ],
            "bg": "#eff6ff",
            "border": "#2563eb",
            "title_color": "#1d4ed8"
        },
        {
            "step": "STEP 2: WEB INTERFACE & REQUEST ORCHESTRATION",
            "title": "Flask Web Application Server (app.py:5000)",
            "details": [
                "• HTTP POST /predict controller with secure file handling and UUID assignment",
                "• Asynchronously parses numerical health profile and dispatches ECG image to digitization pipeline"
            ],
            "bg": "#faf5ff",
            "border": "#9333ea",
            "title_color": "#7e22ce"
        },
        {
            "step": "STEP 3: MULTI-STREAM PREPROCESSING & DIGITIZATION",
            "title": "Clinical Normalization & OpenCV 12-Lead ECG Extraction",
            "details": [
                "• Clinical Normalization: Bounded feature scaling to [0, 1] interval and physical activity mapping",
                "• OpenCV ECG Digitization: 3×4 crop, HSV dual-red masking, 2×2 morphological filtering,",
                "  scanline centroid tracking, linear interpolation, zero-mean centering, and resize to (1000, 12)"
            ],
            "bg": "#f0fdf4",
            "border": "#16a34a",
            "title_color": "#15803d"
        },
        {
            "step": "STEP 4: CLINICAL SCORING & FEATURE PROCESSING",
            "title": "Deterministic Linear Weighted Heuristic Engine",
            "details": [
                "• S_clinical = 0.15·Age + 0.20·BP + 0.20·Chol + 0.10·BMI + 0.15·Diab + 0.10·Smok + 0.05·Fam + 0.05·Act",
                "• Deterministic rule-based scoring yielding bounded baseline risk S_clinical ∈ [0.0, 1.0] (No Tabular ML)"
            ],
            "bg": "#fffbeb",
            "border": "#d97706",
            "title_color": "#b45309"
        },
        {
            "step": "STEP 5: DEEP LEARNING ECG DIAGNOSTIC INFERENCE",
            "title": "1D Deep Convolutional Neural Network (444,101 Parameters)",
            "details": [
                "• 4 Conv Blocks (Conv1D 64/128/256/256, BatchNorm, MaxPool1D, Dropout) + GAP + Dense Head",
                "• 5 Multi-Label Sigmoid Outputs: NORM, MI, STTC, HYP, CD  |  Confidence: C_ECG = max(P_ECG)"
            ],
            "bg": "#fefce8",
            "border": "#ca8a04",
            "title_color": "#a16207"
        },
        {
            "step": "STEP 6: DECISION-LEVEL MULTIMODAL FUSION",
            "title": "Heart Disease Risk Assessment Agent (Fusion Engine)",
            "details": [
                "• Linear Convex Combination: S_combined = (0.40 · S_clinical) + (0.60 · C_ECG)",
                "• Dynamic Clamping: S_combined = clip(S_combined, 0.05, 0.98)  |  Risk Percentage = S_combined × 100%"
            ],
            "bg": "#fff1f2",
            "border": "#e11d48",
            "title_color": "#be123c"
        },
        {
            "step": "STEP 7: RISK STRATIFICATION & EXPLAINABILITY",
            "title": "Decision Boundary Tiers & Matplotlib Attribution Visualizer",
            "details": [
                "• Clinical Risk Stratification: Low (< 0.35), Medium (0.35 – 0.70), High (≥ 0.70)",
                "• Factor Attribution: Relative contribution chart (BP, Chol, Age, BMI, Smoking, Fam History)"
            ],
            "bg": "#f0fdfa",
            "border": "#0d9488",
            "title_color": "#0f766e"
        },
        {
            "step": "STEP 8: PATIENT RESULTS & HEALTH AWARENESS",
            "title": "Interactive Client Dashboard (result.html)",
            "details": [
                "• Overall Risk Badge & Percentage, Primary ECG Diagnostic Classification & DL Confidence",
                "• 5-Class Probability Progress Bars, Embedded Factor Attribution Chart, 8 Preventive Health Practices"
            ],
            "bg": "#f8fafc",
            "border": "#475569",
            "title_color": "#1e293b"
        }
    ]

    # Layout dimensions
    n_steps = len(steps)
    box_w = 88.0
    box_h = 8.4
    x_left = 6.0
    y_start = 86.5
    gap = 2.3

    # Draw Title Header
    ax.text(50.0, 98.8, "PROPOSED SYSTEM WORKFLOW", ha="center", va="center", 
            fontsize=13.5, fontweight="bold", color="#0f172a")
    ax.text(50.0, 97.3, "End-to-End Operational Pipeline of the Heart Disease Risk Assessment Agent", 
            ha="center", va="center", fontsize=8.6, fontstyle="italic", color="#475569")

    # Render each workflow box
    for i, s in enumerate(steps):
        curr_y = y_start - i * (box_h + gap)

        # Outer rounded box
        box = FancyBboxPatch((x_left, curr_y), box_w, box_h, boxstyle="round,pad=0,rounding_size=0.8",
                             facecolor=s["bg"], edgecolor=s["border"], linewidth=1.3, zorder=2)
        ax.add_patch(box)

        # Step Pill Badge on Left
        pill_w = 44.0
        pill_h = 1.9
        pill = FancyBboxPatch((x_left + 1.2, curr_y + box_h - 2.3), pill_w, pill_h,
                              boxstyle="round,pad=0,rounding_size=0.4",
                              facecolor=s["border"], edgecolor="none", zorder=3)
        ax.add_patch(pill)
        ax.text(x_left + 1.2 + pill_w / 2.0, curr_y + box_h - 1.35, s["step"],
                ha="center", va="center", fontsize=7.2, fontweight="bold", color="#ffffff", zorder=4)

        # Subtitle / Component name
        ax.text(x_left + pill_w + 2.5, curr_y + box_h - 1.35, s["title"],
                ha="left", va="center", fontsize=8.2, fontweight="bold", color=s["title_color"], zorder=4)

        # Detail lines
        line_y = curr_y + box_h - 3.7
        for line in s["details"]:
            ax.text(x_left + 2.5, line_y, line, ha="left", va="center", fontsize=7.4, color="#1e293b", zorder=4)
            line_y -= 1.8

        # Downward Arrow to next step
        if i < n_steps - 1:
            arrow_start_y = curr_y
            arrow_end_y = curr_y - gap
            ax.annotate("", xy=(50.0, arrow_end_y), xytext=(50.0, arrow_start_y),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.6, mutation_scale=12),
                        zorder=5)

    # Caption
    ax.text(50.0, 1.8, "Figure 2. End-to-end operational workflow of the proposed Heart Disease Risk Assessment Agent,", 
            ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0f172a")
    ax.text(50.0, 0.7, "depicting the sequential 8-step execution flow from multi-modal input acquisition to final risk stratification and interactive dashboard presentation.", 
            ha="center", va="center", fontsize=7.8, fontstyle="italic", color="#475569")

    plt.tight_layout()

    # Save outputs
    png_path = "workflow_diagram.png"
    pdf_path = "workflow_diagram.pdf"
    
    print("Saving workflow_diagram.png (400 DPI)...")
    plt.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="#ffffff", edgecolor="none")
    
    print("Saving workflow_diagram.pdf (Vector)...")
    plt.savefig(pdf_path, dpi=400, bbox_inches="tight", facecolor="#ffffff", edgecolor="none")

    plt.close()
    print("Workflow diagram generated successfully.")

if __name__ == "__main__":
    create_proposed_workflow_diagram()
