import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

def create_system_architecture_diagram():
    # Set publication-standard font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['font.size'] = 9.0
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # Canvas dimensions: 18.0 x 26.0 inches
    fig_w, fig_h = 18.0, 26.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # Palette - Strict IEEE Medical Informatics Style
    C_LAYER_BG = '#f8fafc'
    C_LAYER_BORDER = '#94a3b8'
    C_PILL_BG = '#0f172a'
    
    C_BLUE_BG = '#eff6ff'
    C_BLUE_BORDER = '#2563eb'
    C_BLUE_TITLE = '#1e40af'
    
    C_GREEN_BG = '#f0fdf4'
    C_GREEN_BORDER = '#16a34a'
    C_GREEN_TITLE = '#15803d'
    
    C_PURPLE_BG = '#faf5ff'
    C_PURPLE_BORDER = '#9333ea'
    C_PURPLE_TITLE = '#6b21a8'
    
    C_AMBER_BG = '#fffbeb'
    C_AMBER_BORDER = '#d97706'
    C_AMBER_TITLE = '#92400e'
    
    C_ROSE_BG = '#fff1f2'
    C_ROSE_BORDER = '#e11d48'
    C_ROSE_TITLE = '#9f1239'
    
    C_TEAL_BG = '#f0fdfa'
    C_TEAL_BORDER = '#0d9488'
    C_TEAL_TITLE = '#115e59'
    
    C_SLATE_BG = '#f8fafc'
    C_SLATE_BORDER = '#475569'
    C_SLATE_TITLE = '#1e293b'

    C_TEXT_DARK = '#0f172a'
    C_TEXT_MUTED = '#475569'
    C_ARROW = '#334155'

    # Helper: Draw Layer Container
    def draw_layer(y, h, num_str, title, subtitle=""):
        layer_box = FancyBboxPatch((3.0, y), 94.0, h, boxstyle="round,pad=0,rounding_size=0.9",
                                   facecolor=C_LAYER_BG, edgecolor=C_LAYER_BORDER, linewidth=1.1, linestyle='--', zorder=1)
        ax.add_patch(layer_box)
        
        # Pill Tag
        pill_w = 32.0 if "&" in num_str else 26.0
        pill = FancyBboxPatch((3.8, y + h - 2.1), pill_w, 1.9, boxstyle="round,pad=0,rounding_size=0.4",
                              facecolor=C_PILL_BG, edgecolor='none', zorder=2)
        ax.add_patch(pill)
        ax.text(3.8 + pill_w/2.0, y + h - 1.15, f"{num_str}: {title.upper()}", 
                ha='center', va='center', fontsize=8.0, fontweight='bold', color='#ffffff', zorder=3)
        
        if subtitle:
            ax.text(95.5, y + h - 1.15, subtitle, ha='right', va='center', fontsize=7.8, 
                    fontstyle='italic', color='#64748b', zorder=3)

    # Helper: Draw Component Box
    def draw_box(x, y, w, h, bg, border, title="", subtitle="", lines=None, 
                 radius=0.7, title_color=None, align='left', title_size=9.2, body_size=7.6, line_spacing=1.05):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg, edgecolor=border, linewidth=1.25, zorder=2)
        ax.add_patch(box)
        
        tc = title_color if title_color else border
        curr_y = y + h - 1.25
        
        if title:
            tx = x + w/2.0 if align == 'center' else x + 1.4
            ax.text(tx, curr_y, title, ha=align, va='center', fontsize=title_size, fontweight='bold', color=tc, zorder=4)
            curr_y -= 1.45
            
        if subtitle:
            sx = x + w/2.0 if align == 'center' else x + 1.4
            ax.text(sx, curr_y, subtitle, ha=align, va='center', fontsize=body_size, fontstyle='italic', color=C_TEXT_MUTED, zorder=4)
            curr_y -= 1.25
            
        if lines:
            for item in lines:
                if item == "":
                    curr_y -= 0.5
                    continue
                lx = x + w/2.0 if align == 'center' else x + 1.4
                ax.text(lx, curr_y, item, ha=align, va='center', fontsize=body_size, color=C_TEXT_DARK, zorder=4)
                curr_y -= line_spacing

    # Helper: Draw Directional Arrow
    def draw_arrow(x1, y1, x2, y2, label="", label_pos='right', color=C_ARROW, lw=1.3, rad=0.0):
        cstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, 
                                    mutation_scale=10, connectionstyle=cstyle),
                    zorder=5)
        if label:
            mx, my = (x1 + x2)/2.0, (y1 + y2)/2.0
            if label_pos == 'right':
                ax.text(mx + 0.7, my, label, ha='left', va='center', fontsize=7.3, fontweight='bold', color=color, zorder=6)
            elif label_pos == 'left':
                ax.text(mx - 0.7, my, label, ha='right', va='center', fontsize=7.3, fontweight='bold', color=color, zorder=6)
            elif label_pos == 'top':
                ax.text(mx, my + 0.5, label, ha='center', va='bottom', fontsize=7.3, fontweight='bold', color=color, zorder=6)
            elif label_pos == 'bottom':
                ax.text(mx, my - 0.5, label, ha='center', va='top', fontsize=7.3, fontweight='bold', color=color, zorder=6)

    # =========================================================================
    # HEADER
    # =========================================================================
    ax.text(50.0, 99.0, "SYSTEM ARCHITECTURE: HEART DISEASE RISK ASSESSMENT AGENT", 
            ha='center', va='center', fontsize=14.0, fontweight='bold', color='#0f172a')
    ax.text(50.0, 97.8, "End-to-End Multimodal Framework: OpenCV 12-Lead ECG Digitization, 1D Deep CNN & Deterministic Clinical Scoring", 
            ha='center', va='center', fontsize=8.8, fontstyle='italic', color='#475569')

    # =========================================================================
    # LAYER 1: CLIENT & INPUT ACQUISITION LAYER (y: 87.8 to 96.6, h=8.8)
    # =========================================================================
    draw_layer(87.8, 8.8, "LAYER 1", "Input Acquisition Layer", "Client Web Browser (details.html)")
    
    draw_box(4.5, 88.4, 43.5, 6.6, C_BLUE_BG, C_BLUE_BORDER, 
             title="Clinical Health Profile Inputs (8 Active Features)", 
             title_color=C_BLUE_TITLE, align='left',
             lines=[
                 "• Demographics: Age (years), Gender (collected in web form; non-scored)",
                 "• Biomarkers: Systolic BP (mmHg), Total Cholesterol (mg/dL), BMI (kg/m²)",
                 "• Medical History: Diabetes (0/1), Smoking (0/1), Family History (0/1)",
                 "• Lifestyle: Physical Exercise Level (0 = Low, 1 = Moderate, 2 = High)"
             ], body_size=7.4, line_spacing=1.0)

    draw_box(52.0, 88.4, 43.5, 6.6, C_GREEN_BG, C_GREEN_BORDER, 
             title="12-Lead ECG Printout Image Acquisition", 
             title_color=C_GREEN_TITLE, align='left',
             lines=[
                 "• Supported Image Formats: Raster Image (.jpg, .jpeg, .png)",
                 "• Standard Clinical Layout: 3×4 printed grid (I, II, III, aVR, aVL, aVF, V1-V6)",
                 "• Acquisition Method: Flatbed scanned paper tracing / Mobile camera capture",
                 "• Input Validation: File existence & extension verification (allowed_file)"
             ], body_size=7.4, line_spacing=1.0)

    # Arrows L1 -> L2
    draw_arrow(26.25, 88.4, 26.25, 85.8, "Multipart Form Submission", label_pos='left')
    draw_arrow(73.75, 88.4, 73.75, 85.8, "Binary Image Upload", label_pos='right')

    # =========================================================================
    # LAYER 2: WEB APPLICATION SERVER & PIPELINE DISPATCHER (y: 77.8 to 85.8, h=8.0)
    # =========================================================================
    draw_layer(77.8, 8.0, "LAYER 2", "Server & Pipeline Dispatcher", "Flask Web Framework (app.py:5000)")
    
    draw_box(4.5, 78.4, 91.0, 5.6, C_PURPLE_BG, C_PURPLE_BORDER,
             title="Flask HTTP Web Application Server & Request Orchestration (app.py)",
             title_color=C_PURPLE_TITLE, align='center',
             lines=[
                 "HTTP POST /predict Controller | werkzeug.utils.secure_filename & UUID Image Storage in static/uploads/",
                 "Asynchronous Request Parsing: Extracts 8 Numerical Clinical Values & Validates Uploaded ECG File Format",
                 "Pipeline Dispatcher: Directs Clinical Values to Normalizer and ECG File to OpenCV Digitizer & 1D CNN"
             ], body_size=7.5, line_spacing=1.1)

    # Arrows L2 -> L3
    draw_arrow(25.0, 78.4, 25.0, 75.8, "8 Clinical Parameters", label_pos='left')
    draw_arrow(72.0, 78.4, 72.0, 75.8, "Raw ECG Image File", label_pos='right')

    # =========================================================================
    # LAYER 3: PREPROCESSING & IMAGE DIGITIZATION (y: 63.2 to 75.8, h=12.6)
    # =========================================================================
    draw_layer(63.2, 12.6, "LAYER 3", "Preprocessing & Image Digitization", "Deterministic Normalization & OpenCV Module")
    
    draw_box(4.5, 63.8, 39.5, 10.2, C_BLUE_BG, C_BLUE_BORDER,
             title="Clinical Feature Normalization Engine",
             title_color=C_BLUE_TITLE, align='left',
             lines=[
                 "Bounded Scaling to [0.0, 1.0] Interval:",
                 "  • S_age  = min(max(Age / 100.0, 0), 1.0)",
                 "  • S_bp   = min(max(BP / 200.0, 0), 1.0)",
                 "  • S_chol = min(max(Cholesterol / 300.0, 0), 1.0)",
                 "  • S_bmi  = min(max(BMI / 40.0, 0), 1.0)",
                 "  • S_diab, S_smok, S_fam = 1.0 if Val > 0 else 0.0",
                 "  • S_act  = 1.0 (Low), 0.5 (Moderate), 0.1 (High)"
             ], body_size=7.3, line_spacing=0.95)

    draw_box(46.5, 63.8, 49.0, 10.2, C_GREEN_BG, C_GREEN_BORDER,
             title="OpenCV 12-Lead ECG Digitization Pipeline (ecg_image_processor.py)",
             title_color=C_GREEN_TITLE, align='left',
             lines=[
                 "1. Grid Partitioning: 3×4 cell crop (margin_x=3%, margin_y=8%, top=15%, bot=92%)",
                 "2. Dual-Threshold HSV Masking: Red grid filter ([0,60,40]-[15,255,255] | [165,60,40]-[180,255,255])",
                 "3. Grayscale Ink Thresholding: Dark trace extraction (intensity 20–110) + bitwise OR merge",
                 "4. Morphological Filtering: 2×2 structuring element Open & Close (speckle noise removal)",
                 "5. Vertical Scanline Tracing: Centroid column tracking + 1D linear interpolation across gaps",
                 "6. Standardization: Inversion (-s), zero-mean centering, unit-peak scaling to [-1.0, 1.0]",
                 "7. Resampling & Lead Stacking: cv2.resize to 1,000 pts/lead → Output Array: (1000, 12)"
             ], body_size=7.2, line_spacing=0.95)

    # Arrows L3 -> L4 and L3 -> L5
    draw_arrow(24.25, 63.8, 24.25, 61.2, "Normalized Parameters", label_pos='left')
    draw_arrow(71.0, 63.8, 71.0, 61.2, "12-Lead Digitized Tensor (1000, 12)", label_pos='right')

    # =========================================================================
    # LAYER 4 & 5: DUAL INFERENCE ENGINES (y: 47.0 to 61.2, h=14.2)
    # =========================================================================
    draw_layer(47.0, 14.2, "LAYERS 4 & 5", "Dual Diagnostic Inference Engines", "Layer 4: Clinical Weighted Sum | Layer 5: 1D Deep CNN")

    # Layer 4 Box (Clinical Weighted Sum)
    draw_box(4.5, 47.6, 39.5, 11.8, C_BLUE_BG, C_BLUE_BORDER,
             title="Layer 4: Deterministic Clinical Score Engine",
             subtitle="calculate_clinical_score() in app.py",
             title_color=C_BLUE_TITLE, align='left',
             lines=[
                 "Deterministic Linear Weighted Sum Formula:",
                 "  S_clinical = 0.15 · S_age  + 0.20 · S_bp",
                 "             + 0.20 · S_chol + 0.10 · S_bmi",
                 "             + 0.15 · S_diab + 0.10 · S_smok",
                 "             + 0.05 · S_fam  + 0.05 · S_act",
                 "",
                 "• Range: S_clinical = clip(score, 0.0, 1.0)",
                 "• Deterministic Medical Heuristic (No Tabular ML)",
                 "• Direct Feature Factor Attribution"
             ], body_size=7.3, line_spacing=0.95)

    # Layer 5 Box (1D Deep CNN)
    draw_box(46.5, 47.6, 49.0, 11.8, C_AMBER_BG, C_AMBER_BORDER,
             title="Layer 5: Deep 1D Convolutional Neural Network",
             subtitle="models/ecg_cnn_improved_best.keras (444,101 Total Parameters)",
             title_color=C_AMBER_TITLE, align='left',
             lines=[
                 "Input Shape: (1000, 12) Digitized 12-Lead ECG Signal",
                 "• Block 1: Conv1D(64, k=7, relu) → BatchNorm → MaxPool1D(2) → Dropout(0.15)",
                 "• Block 2: Conv1D(128, k=5, relu) → BatchNorm → MaxPool1D(2) → Dropout(0.20)",
                 "• Block 3: Conv1D(256, k=3, relu) → BatchNorm → MaxPool1D(2) → Dropout(0.25)",
                 "• Block 4: Conv1D(256, k=3, relu) → BatchNorm → GlobalAveragePooling1D()",
                 "• Dense Head: Dense(256, relu) → Dropout(0.35) → Dense(128, relu) → Dropout(0.25)",
                 "• Output Layer: Dense(5, Sigmoid) → Multi-Label Probabilities P_ECG",
                 "  Classes: NORM (Normal), MI (Infarction), STTC (ST/T), HYP (Hypertrophy), CD (Conduction)",
                 "  Param Breakdown: 442,693 Trainable | 1,408 Non-Trainable (BatchNorm)"
             ], body_size=7.1, line_spacing=0.92)

    # Arrows L4/L5 -> L6
    draw_arrow(24.25, 47.6, 35.0, 44.8, "S_clinical (Weight: 40%)", label_pos='left', rad=-0.08)
    draw_arrow(71.0, 47.6, 60.0, 44.8, "C_ECG & Class Probabilities (Weight: 60%)", label_pos='right', rad=0.08)

    # =========================================================================
    # LAYER 6: MULTIMODAL DECISION FUSION (y: 36.4 to 44.8, h=8.4)
    # =========================================================================
    draw_layer(36.4, 8.4, "LAYER 6", "Multimodal Decision Fusion", "Linear Convex Combination & Dynamic Range Clamping")
    
    draw_box(4.5, 37.0, 91.0, 6.0, C_ROSE_BG, C_ROSE_BORDER,
             title="Multimodal Decision Fusion & Confidence Extraction Engine (app.py)",
             title_color=C_ROSE_TITLE, align='center',
             lines=[
                 "1. ECG Confidence Extraction: C_ECG = max(P_NORM, P_MI, P_STTC, P_HYP, P_CD)  |  Primary ECG Class = argmax(P_ECG)",
                 "2. Linear Weighted Convex Fusion: S_combined = (0.40 · S_clinical) + (0.60 · C_ECG)",
                 "3. Dynamic Safety Clamping: S_combined = clip(S_combined, 0.05, 0.98)  |  Display Score Percentage = S_combined × 100%"
             ], body_size=7.5, line_spacing=1.1)

    # Arrows L6 -> L7
    draw_arrow(36.0, 37.0, 26.5, 34.4, "S_combined Risk Score", label_pos='left', rad=-0.06)
    draw_arrow(64.0, 37.0, 73.5, 34.4, "Clinical Vector & Risk Factors", label_pos='right', rad=0.06)

    # =========================================================================
    # LAYER 7: RISK STRATIFICATION & EXPLAINABILITY (y: 22.8 to 34.4, h=11.6)
    # =========================================================================
    draw_layer(22.8, 11.6, "LAYER 7", "Risk Stratification & Explainability", "Decision Boundaries & Matplotlib Chart Generation")

    # Risk Tiers Box
    draw_box(4.5, 23.4, 43.5, 9.2, C_TEAL_BG, C_TEAL_BORDER,
             title="Clinical Risk Stratification Engine",
             title_color=C_TEAL_TITLE, align='left',
             lines=[
                 "Deterministic Risk Tier Classification:",
                 "  • Low Risk Tier:      S_combined < 0.35 (< 35.0%)",
                 "  • Medium Risk Tier: 0.35 ≤ S_combined < 0.70 (35.0% – 70.0%)",
                 "  • High Risk Tier:     S_combined ≥ 0.70 (≥ 70.0%)",
                 "",
                 "Categorical Tagging: 'Low Risk' | 'Medium Risk' | 'High Risk'"
             ], body_size=7.4, line_spacing=1.0)

    # Explainability Box
    draw_box(52.0, 23.4, 43.5, 9.2, C_TEAL_BG, C_TEAL_BORDER,
             title="Explainability & Factor Attribution Visualizer",
             title_color=C_TEAL_TITLE, align='left',
             lines=[
                 "Matplotlib Feature Contribution Generation (generate_risk_factor_chart):",
                 "  • Normalized Relative Factors: BP/180.0, Chol/300.0, Age/100.0,",
                 "    BMI/40.0, Smoking (0/1), Family History (0/1)",
                 "  • Visual Artifact: Horizontal Bar Chart → static/uploads/risk_factors_<uuid>.png",
                 "  • Response Property: shap_image_url returned in JSON payload"
             ], body_size=7.4, line_spacing=1.0)

    # Arrows L7 -> L8
    draw_arrow(26.25, 23.4, 38.0, 20.8, "Risk Category & Percentage Score", label_pos='left', rad=-0.05)
    draw_arrow(73.75, 23.4, 62.0, 20.8, "Diagnostic Probabilities & Chart URL", label_pos='right', rad=0.05)

    # =========================================================================
    # LAYER 8: CLIENT PRESENTATION & DASHBOARD LAYER (y: 6.2 to 20.8, h=14.6)
    # =========================================================================
    draw_layer(6.2, 14.6, "LAYER 8", "Client Presentation & Clinical Awareness", "Web Frontend Result Dashboard (result.html)")

    draw_box(4.5, 6.8, 91.0, 12.2, C_SLATE_BG, C_SLATE_BORDER,
             title="Comprehensive Patient Results & Health Awareness Dashboard (result.html)",
             subtitle="Asynchronously hydrated via sessionStorage JSON payload from /predict",
             title_color=C_SLATE_TITLE, align='left',
             lines=[
                 "Dashboard Component Architecture:",
                 "1. Overall Risk Banner: Prominent Risk Category Header ('Low Risk' / 'Medium Risk' / 'High Risk') & Combined Percentage Score (S_combined × 100%)",
                 "2. ECG Diagnostic Analysis Panel: Predicted Primary Diagnostic Class (e.g. MI, NORM) & Deep Learning Confidence Percentage (C_ECG × 100%)",
                 "3. Multi-Class Probability Distribution: Individual Dynamic Visual Progress Bars for all 5 Diagnostic Classes (NORM, MI, STTC, HYP, CD)",
                 "4. Explainable AI Visualization Panel: Embedded High-Resolution Factor Attribution Chart (shap_image_url)",
                 "5. Evidence-Based Health Awareness Guidelines: 8 Preventive Lifestyle Practices (Physical activity, balanced diet, smoking cessation, BP/cholesterol monitoring, sleep, stress management, hydration, physician consultation)",
                 "6. Safety & Non-Diagnostic Disclaimer: Clear clinical research notice that assessment is AI-assisted and not a replacement for medical diagnosis"
             ], body_size=7.4, line_spacing=1.2)

    # =========================================================================
    # FIGURE CAPTION
    # =========================================================================
    ax.text(50.0, 3.8, "Figure 1. End-to-end system architecture of the proposed Heart Disease Risk Assessment Agent.", 
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0f172a')
    ax.text(50.0, 2.2, "Illustrating the 8-layer multi-modal workflow from client clinical profile acquisition and OpenCV 12-lead ECG image digitization to 1D Deep CNN inference, deterministic clinical scoring, decision-level fusion, and patient result presentation.", 
            ha='center', va='center', fontsize=8.4, fontstyle='italic', color='#475569')

    plt.tight_layout()
    
    png_path = "system_architecture.png"
    pdf_path = "system_architecture.pdf"
    
    print("Saving system_architecture.png (400 DPI)...")
    plt.savefig(png_path, dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none')
    
    print("Saving system_architecture.pdf (Vector)...")
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight', facecolor='#ffffff', edgecolor='none')
    
    plt.close()
    print("Diagram generated successfully!")

if __name__ == "__main__":
    create_system_architecture_diagram()
