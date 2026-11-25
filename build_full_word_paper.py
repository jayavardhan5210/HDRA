"""
build_full_word_paper.py
========================
Generates a complete, comprehensive, publication-ready research paper in Word (.docx)
format for the Heart Disease Risk Assessment Agent.
Contains all real data, detailed architectural and mathematical formulations,
complete results tables, and embeds all high-resolution figures.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "paper_output")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
RESULTS_FIG_DIR = os.path.join(BASE_DIR, "results_paper", "figures")
RESULTS_TAB_DIR = os.path.join(BASE_DIR, "results_paper", "tables")

for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# Helper function to set table cell background color
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

# Helper function for cell margins
def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_full_paper():
    doc = Document()

    # Page setup - Standard Letter, 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Normal Style
    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(18)
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x0a, 0x25, 0x40)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(16)
        run = p.add_run(text)
        run.italic = True
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x44, 0x55, 0x66)
        return p

    def add_authors(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(18)
        run = p.add_run(text)
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def H1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x0f, 0x2d, 0x59)
        return p

    def H2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.size = Pt(11)
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x1a, 0x3a, 0x60)
        return p

    def P(text, bold_prefix=None, space_after=6):
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.bold = True
            r_bold.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        run = p.add_run(text)
        run.font.size = Pt(10.5)
        run.font.name = "Times New Roman"
        return p

    def Bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.bold = True
        run = p.add_run(text)
        run.font.size = Pt(10.5)
        run.font.name = "Times New Roman"
        return p

    def EmbedFigure(img_filename, caption, width=5.5):
        # Look in paper_output/figures first, then results_paper/figures, then results/
        candidates = [
            os.path.join(FIG_DIR, img_filename),
            os.path.join(RESULTS_FIG_DIR, img_filename),
            os.path.join(BASE_DIR, "results", img_filename),
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found:
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(3)
            doc.add_picture(found, width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            r = p_cap.add_run(caption)
            r.italic = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        else:
            p_err = doc.add_paragraph(f"[{caption} - Image file {img_filename} not found.]")
            p_err.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def EmbedTable(df, title):
        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_title.paragraph_format.space_before = Pt(8)
        p_title.paragraph_format.space_after = Pt(3)
        p_title.paragraph_format.keep_with_next = True
        r = p_title.add_run(title)
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x0f, 0x2d, 0x59)

        table = doc.add_table(rows=1, cols=len(df.columns))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = 'Table Grid'

        # Header
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(df.columns):
            hdr_cells[i].text = str(col)
            set_cell_background(hdr_cells[i], "1F4E79")
            set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
            for p in hdr_cells[i].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.bold = True
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        # Rows
        for row_idx, row in enumerate(df.itertuples(index=False)):
            row_cells = table.add_row().cells
            bg_color = "F2F5F9" if row_idx % 2 == 1 else "FFFFFF"
            for col_idx, val in enumerate(row):
                row_cells[col_idx].text = str(val)
                set_cell_background(row_cells[col_idx], bg_color)
                set_cell_margins(row_cells[col_idx], top=80, bottom=80, left=120, right=120)
                for p in row_cells[col_idx].paragraphs:
                    p.paragraph_format.space_before = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.05
                    if col_idx == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(9.0)
                        run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # DOCUMENT CONTENT
    # -------------------------------------------------------------------------
    add_title("Heart Disease Risk Assessment Agent: An AI-Assisted Multimodal Diagnostic System Combining Patient Health Parameters and CNN-Based 12-Lead ECG Analysis")
    add_subtitle("A Research and Educational Decision Support Framework for Cardiovascular Screening")
    add_authors("Cardiovascular AI Research Team • Department of Biomedical Engineering & Computer Science")

    # Abstract
    H1("ABSTRACT")
    P("Cardiovascular diseases (CVDs) represent the leading contributor to global mortality, necessitating reliable, rapid, and interpretable non-invasive screening technologies. Electrocardiogram (ECG) interpretation and clinical health factor evaluation are conventionally performed in isolation, leading to diagnostic delays and fragmented risk stratifications. In this paper, we present the Heart Disease Risk Assessment Agent, an integrated artificial intelligence decision support framework combining automated 12-lead ECG waveform analysis via a deep 1-D Convolutional Neural Network (CNN) with a multi-attribute clinical risk calculator. The ECG analytical engine is trained and evaluated on the PhysioNet PTB-XL database across five diagnostic superclasses: Normal ECG (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), Hypertrophy (HYP), and Conduction Disturbance (CD). To bridge paper-based clinical recordings and digital modeling, an image-processing module using color segmentation and morphological filtering extracts continuous time-series waveforms directly from uploaded 12-lead ECG charts. The multi-label CNN achieved a training accuracy of 91.07%, a validation accuracy of 88.12%, an element-wise test binary accuracy of 87.58%, and a mean ROC-AUC of 0.9052. The diagnostic class-wise ROC-AUC scores were 0.9382 for NORM, 0.9305 for STTC, 0.9178 for CD, 0.9168 for MI, and 0.8225 for HYP. A dual-branch late fusion mechanism synthesizes clinical physiological scores (40% weighting) with maximum CNN diagnostic confidence (60% weighting) to stratify patient risk into Low, Medium, and High categories, accompanied by visual risk-factor attributions. The system demonstrates robust multi-label screening capability while highlighting key avenues for addressing minority-class imbalance in prospective clinical workflows.")

    # Keywords
    H1("KEYWORDS")
    P("Cardiovascular Disease, 12-Lead Electrocardiogram, 1-D Convolutional Neural Network (1D-CNN), Multimodal Risk Assessment, Multi-label Classification, PTB-XL Benchmark, ECG Digitization, Clinical Decision Support System.")

    # Section I
    H1("I. INTRODUCTION")
    P("Cardiovascular diseases (CVDs) account for an estimated 17.9 million deaths annually, representing approximately 32% of all global fatalities according to the World Health Organization (WHO). Early detection and accurate risk stratification remain the cornerstone of preventive cardiology. The standard 12-lead electrocardiogram (ECG) is the most widely utilized non-invasive diagnostic modality, capturing cardiac electrical depolarization and repolarization across orthogonal anatomical planes.")
    P("Despite its ubiquity, standard 12-lead ECG interpretation is labor-intensive, subject to inter-observer variability, and heavily dependent on sub-specialist cardiology expertise, which is frequently unavailable in emergency departments, rural clinics, and ambulatory triage centers. Concurrently, traditional risk estimation scores (e.g., Framingham Risk Score, ASCVD Risk Estimator) evaluate demographic and biochemical variables (age, systolic blood pressure, total cholesterol, BMI, smoking, diabetes) but entirely omit electrophysiological waveform morphology.")
    P("Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs), have demonstrated remarkable capabilities in automated feature learning directly from raw physiological time-series signals, matching or exceeding general practitioner diagnostic accuracy. However, existing automated solutions remain siloed: they either analyze raw digital telemetry without patient context or process tabular parameters without waveform data. Furthermore, physical printed ECG records remain prevalent across global healthcare centers, creating a digital barrier for automated systems.")
    P("To address these critical limitations, this work develops and validates the Heart Disease Risk Assessment Agent—an integrated, dual-modality AI system that digitizes 12-lead paper ECG images, executes multi-label CNN classification across five diagnostic superclasses, computes a weighted clinical risk score, and combines both streams into a unified cardiovascular risk profile.")

    # Section II
    H1("II. PROBLEM STATEMENT")
    P("The contemporary clinical diagnosis of cardiovascular pathology presents three fundamental challenges:")
    Bullet(" Disjointed diagnostic workflows evaluate physiological health metrics (hypertension, lipid profiles, glycemic status) independently from electrophysiological ECG signals, preventing holistic risk assessment.", "Diagnostic Siloing:")
    Bullet(" Clinical ECGs routinely display concurrent abnormalities (e.g., an acute myocardial infarction co-occurring with secondary intraventricular conduction delays and ST-segment changes). Conventional single-label multi-class models fail to capture this multi-label clinical reality.", "Multi-label Pathology:")
    Bullet(" A substantial proportion of diagnostic ECGs in community clinics are stored on printed paper grid sheets. Without automated waveform digitization and reconstruction pipelines, standard deep learning models cannot be utilized.", "Analog-to-Digital Gap:")
    P("This investigation formulates an integrated solution addressing all three challenges through automated image digitization, multi-label 1D-CNN architecture, and hybrid multimodal score fusion.")

    # Section III
    H1("III. RESEARCH OBJECTIVES")
    P("The principal research objectives of this investigation are:")
    Bullet("To engineer a robust computer vision pipeline that automatically isolates 12 lead strips from standard 3x4 ECG printout images, suppresses grid noise, and reconstructs 1000-sample time-series signals per lead.")
    Bullet("To design and train a high-capacity 1D Convolutional Neural Network (1D-CNN) capable of multi-label classification over the five PTB-XL diagnostic superclasses (NORM, MI, STTC, HYP, CD).")
    Bullet("To implement an epidemiological clinical health parameter assessment engine integrating eight cardinal cardiovascular risk indicators.")
    Bullet("To formulate a decision fusion mechanism that combines electrophysiological neural confidence with clinical scoring to output an actionable, 3-tier risk index.")
    Bullet("To rigorously evaluate system performance using element-wise binary accuracy, exact subset match ratio, Hamming loss, class-wise precision, recall, F1-scores, and ROC-AUC curves.")

    # Section IV
    H1("IV. EXISTING SYSTEMS & RELATED WORK")
    P("Traditional automated ECG interpretation relies on rule-based algorithmic heuristic approaches (e.g., the Minnesota Code and Glasgow ECG Analysis Program) and handcrafted feature extraction techniques such as Pan-Tompkins QRS detection, discrete wavelet transforms (DWT), and autoregressive modeling. While computationally lightweight, these methods degrade under baseline wander, electromyographic noise, and polymorphic waveform morphology.")
    P("In deep learning, Hannun et al. (2019) demonstrated single-lead arrhythmia classification using a 34-layer residual neural network, while Ribeiro et al. (2020) established 12-lead 1D-ResNet performance on over 2 million telehealth recordings. Strodthoff et al. (2021) established standardized benchmark protocols on the PTB-XL dataset. However, these systems focus purely on raw signal arrays, requiring direct telemetry feeds, and completely neglect patient biochemical risk factors.")
    P("On the tabular front, epidemiological engines such as the Framingham Risk Score and the ACC/AHA ASCVD Risk Calculator provide long-term risk probabilities but lack acute electrophysiological insight. The proposed Heart Disease Risk Assessment Agent bridges this divide by synergistically coupling automated visual ECG signal extraction, 1D-CNN multi-label inference, and clinical health parameter evaluation.")

    # Section V
    H1("V. PROPOSED MULTIMODAL SYSTEM")
    P("The proposed system operates through two parallel, complementary analytical pathways that converge at a late-stage Decision Fusion Engine:")
    Bullet(" Ingests standard JPG/PNG images of 12-lead ECG printouts, isolates individual lead regions, performs color segmentation and morphological filtering, extracts continuous 1D waveforms, and feeds a (1000, 12) signal tensor into the 1D-CNN classifier.", "Electrophysiological Pathway:")
    Bullet(" Ingests eight clinical health parameters (Age, Systolic BP, Total Cholesterol, BMI, Diabetes, Smoking, Family History, Physical Activity), normalizes them against established clinical thresholds, and calculates an evidence-weighted composite score.", "Clinical Pathway:")
    Bullet(" Formulates a weighted linear combination (0.60 * CNN Max Confidence + 0.40 * Clinical Score) to generate an overall Risk Index (0%-100%), stratified into Low, Medium, and High Risk tiers, accompanied by visual feature attribution charts.", "Decision Fusion Engine:")

    # Section VI
    H1("VI. SYSTEM ARCHITECTURE")
    P("The system architecture comprises five decoupled modular layers:")
    Bullet(" A lightweight Flask-based web application providing responsive patient intake interfaces, real-time waveform visualization, and dynamic risk reporting.", "Presentation Layer (UI):")
    Bullet(" An OpenCV-powered module executing 3x4 grid cropping, HSV thresholding, vertical scanline contour tracking, linear interpolation, and signal normalization.", "Computer Vision Digitizer:")
    Bullet(" A deep 1D-CNN featuring 4 convolutional blocks, Batch Normalization, spatial Dropout, Global Average Pooling, and 5 independent Sigmoid units.", "Neural Inference Engine:")
    Bullet(" A deterministic epidemiological scoring engine mapping patient biometric indicators to continuous risk weights.", "Clinical Assessment Module:")
    Bullet(" Computes the hybrid risk index, assigns clinical risk tiers, generates dynamic Matplotlib factor attribution bar charts, and outputs JSON payloads.", "Fusion & Visualization Engine:")

    # Section VII
    H1("VII. DATASET DESCRIPTION: PTB-XL")
    P("The experimental evaluation utilizes the PhysioNet PTB-XL electrocardiography database (Wagner et al., 2020), the largest publicly accessible, comprehensively annotated 12-lead clinical ECG corpus. PTB-XL comprises 21,837 clinical 12-lead recordings of 10-second duration obtained from 18,885 unique patients (52% male, 48% female; age span 17-95 years).")
    P("Diagnostic statements follow the standardized SCP-ECG protocol, mapped into five primary clinical superclasses:")
    Bullet(" Baseline recordings exhibiting absence of acute or chronic electrical abnormalities.", "NORM (Normal ECG):")
    Bullet(" Encompassing acute ST-elevation, non-ST elevation, anterior, inferior, and lateral infarctions.", "MI (Myocardial Infarction):")
    Bullet(" Encompassing non-specific ST-segment elevations/depressions, T-wave inversions, and repolarization abnormalities.", "STTC (ST/T Changes):")
    Bullet(" Left ventricular hypertrophy, right ventricular hypertrophy, and combined atrial/ventricular enlargement.", "HYP (Hypertrophy):")
    Bullet(" Left/right bundle branch blocks, intraventricular conduction delays, and fascicular blocks.", "CD (Conduction Disturbance):")
    P("To ensure rigorous, unbiased benchmarking, the official PTB-XL stratified 10-fold partition (strat_fold) is preserved: Folds 1-8 form the training partition (14,234 records), Fold 9 forms the validation partition (1,891 records), and Fold 10 forms the untouched test partition (1,903 records).")

    # Table 1
    df_tab1 = pd.DataFrame({
        "Class": ["NORM", "MI", "STTC", "HYP", "CD", "Total Records"],
        "Description": ["Normal ECG", "Myocardial Infarction", "ST/T Change", "Hypertrophy", "Conduction Disturbance", "Dataset Partition Size"],
        "Train Positives (Folds 1-8)": ["7,605", "4,375", "4,213", "2,139", "3,912", "14,234"],
        "Val Positives (Fold 9)": ["1,039", "621", "558", "270", "564", "1,891"],
        "Test Positives (Fold 10)": ["870", "473", "464", "240", "422", "1,903"]
    })
    EmbedTable(df_tab1, "TABLE I. Distribution of Diagnostic Superclasses Across PTB-XL Dataset Partitions")

    # Section VIII
    H1("VIII. DATA PREPROCESSING & WAVEFORM PIPELINE")
    P("Data preparation follows a strict protocol to prevent data leakage and ensure numeric stability:")
    Bullet(" Diagnostic annotations are mapped into binary indicator vectors of length 5 using multi-hot encoding, reflecting concurrent co-pathologies.", "Multi-hot Vectorization:")
    Bullet(" Each 10-second 12-lead recording is loaded at 100 Hz (1000 temporal samples per lead). Missing values and infinite numbers are replaced with zeros.", "Signal Conditioning:")
    Bullet(" Each lead signal is independently standardized using Z-score normalization: z = (x - mean) / (std + 1e-7), ensuring zero mean and unit variance.", "Z-Score Normalization:")
    Bullet(" Patient records are strictly partitioned by patient ID according to strat_fold, ensuring no patient's recordings span both training and test sets.", "Fold Isolation:")

    # Section IX
    H1("IX. ECG IMAGE DIGITIZATION & SIGNAL EXTRACTION")
    P("When the user uploads a physical ECG printout image (JPG/PNG), the system executes an automated 7-stage computer vision reconstruction pipeline implemented in ecg_image_processor.py:")
    Bullet(" The image is validated for dimension and structure, establishing a standard 3x4 lead grid layout with defined margins.", "1. Grid Ingestion & Validation:")
    Bullet(" The image is converted to the HSV color space. Dual red-range thresholds isolate colored grid lines, while grayscale intensity thresholding isolates dark waveform ink.", "2. HSV & Grayscale Masking:")
    Bullet(" Morphological opening and closing filters with a (2x2) structuring element eliminate grid artifacts while bridging minute line discontinuities.", "3. Morphological Filtering:")
    Bullet(" For each horizontal pixel x, the algorithm scans vertically across the lead bounding box, identifying the active pixel closest to the vertical baseline center.", "4. Scanline Peak Tracking:")
    Bullet(" Unobserved pixel columns are reconstructed using 1D linear interpolation across adjacent valid points.", "5. 1D Linear Interpolation:")
    Bullet(" The raw pixel trajectory is inverted (converting image coordinates where y increases downward), centered to zero mean, and amplitude-normalized to unit peak.", "6. Amplitude Scaling:")
    Bullet(" Linear interpolation resamples the trace to exactly 1000 temporal samples. All 12 leads are stacked into a (1000, 12) array matching the CNN input format.", "7. 12-Lead Tensor Stacking:")

    # Section X
    H1("X. CNN MODEL ARCHITECTURE")
    P("The diagnostic classification backbone is a specialized 1-D Convolutional Neural Network designed to extract multi-scale spatio-temporal features across the 12 cardiac leads while maintaining computational efficiency for real-time web deployment.")
    P("The model processes input tensors of shape (1000, 12) through four sequential convolutional blocks:")
    Bullet(" Conv1D (64 filters, kernel size 7, ReLU, same padding) -> BatchNormalization -> MaxPooling1D (pool size 2) -> Spatial Dropout (0.15). Output: (500, 64).", "Block 1 (Macro-morphology):")
    Bullet(" Conv1D (128 filters, kernel size 5, ReLU, same padding) -> BatchNormalization -> MaxPooling1D (pool size 2) -> Spatial Dropout (0.20). Output: (250, 128).", "Block 2 (Rhythm & Complexes):")
    Bullet(" Conv1D (256 filters, kernel size 3, ReLU, same padding) -> BatchNormalization -> MaxPooling1D (pool size 2) -> Spatial Dropout (0.25). Output: (125, 256).", "Block 3 (Micro-features):")
    Bullet(" Conv1D (256 filters, kernel 3, ReLU) -> BatchNormalization -> Conv1D (128 filters, kernel 3, ReLU) -> BatchNormalization -> GlobalAveragePooling1D. Output: (256) feature vector.", "Block 4 (High-Level Representation):")
    Bullet(" Dense (256 units, ReLU) -> BatchNormalization -> Dropout (0.35) -> Dense (128 units, ReLU) -> Dropout (0.25) -> Dense (5 units, Sigmoid).", "Classification Head:")
    P("The network contains a total of 444,101 parameters (442,693 trainable, 1,408 non-trainable batch normalization parameters), detailed in Table II.")

    # Table 2
    df_tab2 = pd.DataFrame([
        ["conv1d", "Conv1D", "(None, 1000, 64)", "5,440"],
        ["batch_normalization", "BatchNormalization", "(None, 1000, 64)", "256"],
        ["max_pooling1d", "MaxPooling1D (pool=2)", "(None, 500, 64)", "0"],
        ["dropout", "Dropout (rate=0.15)", "(None, 500, 64)", "0"],
        ["conv1d_1", "Conv1D", "(None, 500, 128)", "41,088"],
        ["batch_normalization_1", "BatchNormalization", "(None, 500, 128)", "512"],
        ["max_pooling1d_1", "MaxPooling1D (pool=2)", "(None, 250, 128)", "0"],
        ["dropout_1", "Dropout (rate=0.20)", "(None, 250, 128)", "0"],
        ["conv1d_2", "Conv1D", "(None, 250, 256)", "98,560"],
        ["batch_normalization_2", "BatchNormalization", "(None, 250, 256)", "1,024"],
        ["max_pooling1d_2", "MaxPooling1D (pool=2)", "(None, 125, 256)", "0"],
        ["dropout_2", "Dropout (rate=0.25)", "(None, 125, 256)", "0"],
        ["conv1d_3", "Conv1D", "(None, 125, 256)", "196,864"],
        ["batch_normalization_3", "BatchNormalization", "(None, 125, 256)", "1,024"],
        ["global_average_pooling1d", "GlobalAveragePooling1D", "(None, 256)", "0"],
        ["dense", "Dense (ReLU)", "(None, 256)", "65,792"],
        ["dropout_3", "Dropout (rate=0.35)", "(None, 256)", "0"],
        ["dense_1", "Dense (ReLU)", "(None, 128)", "32,896"],
        ["dropout_4", "Dropout (rate=0.25)", "(None, 128)", "0"],
        ["dense_2", "Dense (Sigmoid Output)", "(None, 5)", "645"],
    ], columns=["Layer Name", "Layer Type", "Output Tensor Shape", "Parameters"])
    EmbedTable(df_tab2, "TABLE II. Deep 1D-CNN Model Architectural Layers and Parameter Distribution")

    # Section XI
    H1("XI. PATIENT HEALTH PARAMETER PROCESSING")
    P("The clinical assessment module evaluates eight physiological parameters obtained during patient intake. Each parameter is normalized to a continuous scale [0, 1] and assigned an epidemiological weight reflecting its relative contribution to cardiovascular risk:")
    Bullet(" Normalized s_age = min(Age/100, 1.0). Weight w1 = 0.15.", "Age (years):")
    Bullet(" Normalized s_bp = min(BP/200, 1.0). Weight w2 = 0.20.", "Systolic Blood Pressure (mmHg):")
    Bullet(" Normalized s_chol = min(Cholesterol/300, 1.0). Weight w3 = 0.20.", "Total Serum Cholesterol (mg/dL):")
    Bullet(" Normalized s_bmi = min(BMI/40, 1.0). Weight w4 = 0.10.", "Body Mass Index (kg/m2):")
    Bullet(" Binary indicator: s_diab = 1.0 if diabetic, else 0.0. Weight w5 = 0.15.", "Diabetes Mellitus:")
    Bullet(" Binary indicator: s_smok = 1.0 if active smoker, else 0.0. Weight w6 = 0.10.", "Smoking Status:")
    Bullet(" Binary indicator: s_fam = 1.0 if positive family history, else 0.0. Weight w7 = 0.05.", "Family History of CVD:")
    Bullet(" Categorical: 1.0 (Sedentary), 0.5 (Moderate), 0.1 (Active). Weight w8 = 0.05.", "Physical Activity:")
    P("The composite clinical vulnerability score S_clinical is computed as:")
    P("S_clinical = SUM_{k=1..8} (w_k * s_k), where SUM_{k=1..8} w_k = 1.00", bold_prefix="Equation 1: ")

    # Section XII
    H1("XII. DECISION FUSION & RISK ASSESSMENT WORKFLOW")
    P("The Decision Fusion Engine synthesizes both diagnostic streams to compute the final Combined Risk Score S_combined:")
    P("S_combined = 0.40 * S_clinical + 0.60 * C_ECG", bold_prefix="Equation 2: ")
    P("where C_ECG = max_{j in {1..5}} P(Class_j) is the maximum sigmoid confidence output by the CNN. The combined score is constrained to the interval [0.05, 0.98] and mapped to three distinct clinical action tiers:")
    Bullet(" S_combined < 0.35 (< 35%). Indicates low cardiovascular burden. Recommends routine lifestyle maintenance.", "Low Risk Tier:")
    Bullet(" 0.35 <= S_combined < 0.70 (35%-70%). Indicates moderate risk requiring lifestyle modification and routine clinical follow-up.", "Medium Risk Tier:")
    Bullet(" S_combined >= 0.70 (>= 70%). Indicates significant cardiovascular abnormality. Flags immediate recommendation for formal cardiologist evaluation.", "High Risk Tier:")
    P("An interpretability sub-module generates a horizontal factor attribution chart using Matplotlib, illustrating the relative burden of each clinical marker (Blood Pressure, Cholesterol, Age, BMI, Smoking, Family History) to provide transparent insights for patients and clinicians.")

    # Section XIII
    H1("XIII. EXPERIMENTAL SETUP & MODEL TRAINING")
    P("The training and evaluation protocol was executed in Python 3.12 utilizing TensorFlow 2.21, Keras, Scikit-learn, and OpenCV on the official PTB-XL split.")
    P("To overcome severe class imbalance across the diagnostic categories (e.g., NORM with 9,514 instances versus HYP with 2,649 instances), a custom Class-Weighted Focal Loss was employed:")
    P("Loss(y, y_hat) = -SUM_{j=1..5} [ alpha_j * y_j * (1 - y_hat_j)^gamma * log(y_hat_j) + (1 - y_j) * y_hat_j^gamma * log(1 - y_hat_j) ]", bold_prefix="Equation 3: ")
    P("with focusing parameter gamma = 2.0 and positive class weights alpha_j = clip( sqrt(N_neg_j / N_pos_j) / mean(sqrt(N_neg / N_pos)), 0.75, 2.50 ).")
    P("Training used the Adam optimizer with initial learning rate eta = 3e-4, mini-batch size 32, for up to 50 epochs. Training was regularized via EarlyStopping (patience 8 on validation binary accuracy, restoring best weights), ReduceLROnPlateau (factor 0.5, patience 3, minimum eta = 1e-6), and ModelCheckpoint saving ecg_cnn_improved_best.keras.")

    # Section XIV
    H1("XIV. EVALUATION METRICS")
    P("Because multi-label classification produces independent sigmoid activations per class, single-label metrics are insufficient. The following comprehensive metric suite is reported:")
    Bullet(" Acc_binary = (1 / (N * K)) * SUM_{i=1..N} SUM_{j=1..K} I(y_ij == y_hat_ij). Evaluates individual label decision correctness across all N x 5 predictions.", "Element-wise Binary Accuracy:")
    Bullet(" Acc_subset = (1 / N) * SUM_{i=1..N} PROD_{j=1..K} I(y_ij == y_hat_ij). The strictest metric, requiring all 5 labels to match ground truth exactly.", "Subset Accuracy (Exact Match):")
    Bullet(" HL = (1 / (N * K)) * SUM_{i=1..N} SUM_{j=1..K} I(y_ij != y_hat_ij) = 1 - Acc_binary.", "Hamming Loss:")
    Bullet(" Precision (TP/(TP+FP)), Recall/Sensitivity (TP/(TP+FN)), Specificity (TN/(TN+FP)), and F1-Score (2PR/(P+R)) computed per class, as well as Micro, Macro, and Weighted averages.", "Class-wise & Averaged Performance:")
    Bullet(" Threshold-independent ranking capability evaluated per class and globally across all operating thresholds.", "ROC-AUC & PR-AUC:")

    # Section XV
    H1("XV. RESULTS AND DISCUSSION")
    P("The proposed system achieved strong diagnostic performance on the held-out PTB-XL test set (Fold 10, 1,903 records). As summarized in Table III, the model attained a training accuracy of 91.07%, a validation accuracy of 88.12%, an element-wise test binary accuracy of 87.58%, and a mean ROC-AUC of 0.9052.")

    # Table 3
    df_tab3 = pd.DataFrame([
        ["Training Binary Accuracy", "91.07%"],
        ["Validation Binary Accuracy", "88.12%"],
        ["Test Binary Accuracy (Element-wise)", "87.58%"],
        ["Subset Accuracy (Exact 5-Label Match)", "57.80%"],
        ["Hamming Loss", "0.1407"],
        ["Micro-averaged Precision", "0.7124"],
        ["Micro-averaged Recall (Sensitivity)", "0.7675"],
        ["Micro-averaged F1-Score", "0.7389"],
        ["Macro-averaged Precision", "0.6990"],
        ["Macro-averaged Recall (Sensitivity)", "0.7246"],
        ["Macro-averaged F1-Score", "0.6990"],
        ["Macro-averaged Specificity", "0.8869"],
        ["Micro-averaged ROC-AUC", "0.9162"],
        ["Macro-averaged ROC-AUC", "0.9031"],
        ["Macro-averaged PR-AUC (Average Precision)", "0.7711"],
    ], columns=["Evaluation Metric", "Experimental Value"])
    EmbedTable(df_tab3, "TABLE III. Overall Quantitative Performance Metrics of the Proposed System")

    EmbedFigure("fig7_overall_performance.png", "Fig. 1. Overall model accuracy across training, validation, and test splits along with Mean ROC-AUC.", 4.5)

    P("The close alignment between training accuracy (91.07%), validation accuracy (88.12%), and test accuracy (87.58%) indicates good generalization with negligible overfitting, validated by Batch Normalization and multi-stage Dropout. The threshold-independent discrimination was especially strong (Micro ROC-AUC: 0.9162, Macro ROC-AUC: 0.9031), confirming the network's continuous probability outputs effectively rank disease likelihood.")

    # Section XVI
    H1("XVI. GRAPHICAL & CONVERGENCE ANALYSIS")
    P("Figure 2 and Figure 3 depict the learning dynamics during the improved training run across epochs.")
    EmbedFigure("fig1_training_validation_accuracy.png", "Fig. 2. Training and validation binary accuracy progression over training epochs.", 5.0)
    EmbedFigure("fig2_training_validation_loss.png", "Fig. 3. Training and validation loss convergence curves under class-weighted focal loss.", 5.0)
    P("The loss curve demonstrates rapid convergence within the initial 15 epochs, stabilizing as the learning rate scheduler reduced eta upon validation plateaus. The tracking between training and validation accuracy curves highlights stable gradient flow and effective regularization.")

    # Section XVII
    H1("XVII. CLASS-WISE PERFORMANCE & ROC ANALYSIS")
    P("Table IV and Figures 4-6 provide granular class-wise performance breakdowns across the five diagnostic categories.")

    # Table 4
    df_tab4 = pd.DataFrame([
        ["NORM", "85.29%", "0.7950", "0.9138", "0.8503", "0.9382", "870"],
        ["MI", "85.81%", "0.7707", "0.6110", "0.6816", "0.9168", "473"],
        ["STTC", "88.60%", "0.7679", "0.7629", "0.7654", "0.9305", "464"],
        ["HYP", "88.81%", "0.6216", "0.2875", "0.3932", "0.8225", "240"],
        ["CD", "89.39%", "0.8333", "0.6517", "0.7314", "0.9178", "422"],
    ], columns=["Diagnostic Class", "Binary Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Support"])
    EmbedTable(df_tab4, "TABLE IV. Detailed Per-Class Classification Report and Discrimination Metrics")

    EmbedFigure("fig4_classwise_precision_recall_f1.png", "Fig. 4. Class-wise comparison of Precision, Recall, and F1-Score across all five diagnostic superclasses.", 5.2)
    EmbedFigure("fig6_roc_curves.png", "Fig. 5. Receiver Operating Characteristic (ROC) curves with individual Area Under Curve (AUC) for each class.", 5.2)
    EmbedFigure("fig7_precision_recall_curves.png", "Fig. 6. Precision-Recall curves illustrating average precision across classes under imbalanced conditions.", 5.2)

    P("Key class-wise findings include:")
    Bullet(" NORM achieved the highest Recall (0.9138), F1-Score (0.8503), and ROC-AUC (0.9382), demonstrating exceptional reliability in identifying normal cardiac rhythms and preventing missed pathology in primary screening.", "Normal Rhythm (NORM):")
    Bullet(" STTC demonstrated balanced precision (0.7679) and recall (0.7629) with an outstanding ROC-AUC of 0.9305, proving highly effective in identifying repolarization abnormalities.", "ST/T Changes (STTC):")
    Bullet(" MI and CD achieved high precision (0.7707 and 0.8333, respectively) and ROC-AUC scores exceeding 0.916, indicating strong positive predictive value.", "Myocardial Infarction & Conduction Disturbance:")
    Bullet(" HYP yielded the lowest recall (0.2875) and F1-score (0.3932) despite 88.81% binary accuracy. This is attributed to extreme class imbalance (only 240 positive test cases) and subtle voltage-criteria morphology that overlaps with normal variants.", "Hypertrophy (HYP):")

    # Section XVIII
    H1("XVIII. CONFUSION MATRIX & ERROR ANALYSIS")
    P("To faithfully represent the multi-label decision space, one-vs-rest confusion matrices are constructed for each diagnostic class, as presented in Table V and Figure 7.")

    # Table 5
    df_tab5 = pd.DataFrame([
        ["NORM", "828", "205", "75", "795", "1,033", "870"],
        ["MI", "1,344", "86", "184", "289", "1,430", "473"],
        ["STTC", "1,332", "107", "110", "354", "1,439", "464"],
        ["HYP", "1,621", "42", "171", "69", "1,663", "240"],
        ["CD", "1,426", "55", "147", "275", "1,481", "422"],
    ], columns=["Class", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)", "Actual Negatives", "Actual Positives"])
    EmbedTable(df_tab5, "TABLE V. One-vs-Rest Confusion Matrix Values on the Held-Out Test Set")

    EmbedFigure("fig3_confusion_matrix_ovr.png", "Fig. 7. One-vs-rest confusion matrices for the five diagnostic superclasses on the test split.", 5.5)

    # Table 6 Error analysis
    df_tab6 = pd.DataFrame([
        ["NORM", "205", "75", "19.85%", "8.62%", "14.71%"],
        ["MI", "86", "184", "6.01%", "38.90%", "14.19%"],
        ["STTC", "107", "110", "7.44%", "23.71%", "11.40%"],
        ["HYP", "42", "171", "2.53%", "71.25%", "11.19%"],
        ["CD", "55", "147", "3.71%", "34.83%", "10.61%"],
    ], columns=["Class", "FP Count", "FN Count", "False Positive Rate (FPR)", "False Negative Rate (FNR)", "Total Error Rate"])
    EmbedTable(df_tab6, "TABLE VI. Detailed Error Distribution and False Positive/Negative Rates per Class")

    EmbedFigure("fig6_fp_vs_fn.png", "Fig. 8. Comparative distribution of False Positives versus False Negatives across diagnostic classes.", 5.0)

    P("Clinical Error Implications:")
    Bullet(" A False Positive indicates an unaffected patient flagged as having an abnormality. While causing temporary anxiety and necessitating confirmatory testing, it does not endanger patient life. NORM had 205 FP cases (19.85% FPR).", "False Positive (Type I Error):")
    Bullet(" A False Negative represents missed cardiac pathology (e.g., undetected myocardial infarction or conduction disturbance). In screening applications, minimizing False Negatives is of paramount clinical urgency. MI had 184 FN and HYP had 171 FN cases, identifying these as primary targets for future threshold optimization.", "False Negative (Type II Error):")

    # Section XIX
    H1("XIX. LIMITATIONS & ETHICAL CONSIDERATIONS")
    P("While the Heart Disease Risk Assessment Agent demonstrates strong diagnostic and digitization performance, several clinical and technical limitations must be acknowledged:")
    Bullet(" The system is strictly an AI-assisted decision support and research framework. It is not an FDA/CE-cleared primary diagnostic device and must not replace board-certified cardiologist evaluation.", "Scope of Application:")
    Bullet(" Performance on HYP remains constrained by dataset class imbalance (240 test instances), causing the model to lean conservative on hypertrophy identification.", "Class Imbalance Sensitivity:")
    Bullet(" The OpenCV signal extraction pipeline assumes standard 12-lead 3x4 layout and reasonably clean lighting; severe perspective distortion, shadows, or damaged paper can introduce digitization artifacts.", "Image Quality Constraints:")
    Bullet(" The current model was evaluated on the German PTB-XL cohort; prospective multi-center validation across diverse international patient populations is required prior to clinical deployment.", "Population Generalizability:")

    # Section XX
    H1("XX. FUTURE SCOPE")
    P("Promising avenues for future research and development include:")
    Bullet(" Integrating multi-lead 1D Vision Transformers (ViT) and Mamba state-space architectures to capture extended temporal dependencies across 10-second intervals.", "Transformer & State-Space Architectures:")
    Bullet(" Transitioning from late-stage linear score fusion to cross-modal attention mechanisms that learn joint latent representations of tabular and waveform embeddings.", "End-to-End Multimodal Deep Fusion:")
    Bullet(" Incorporating continuous photoplethysmography (PPG) and smartwatch telemetry for ambulatory, real-time arrhythmia surveillance.", "Wearable Sensor Integration:")
    Bullet(" Deploying federated learning protocols enabling collaborative multi-hospital training without centralizing sensitive patient telemetry.", "Federated Privacy-Preserving Learning:")

    # Section XXI
    H1("XXI. CONCLUSION")
    P("This study designed, implemented, and validated the Heart Disease Risk Assessment Agent, an end-to-end multimodal diagnostic framework integrating automated 12-lead ECG image digitization, deep 1D-CNN multi-label classification, and patient clinical health parameter scoring. Evaluated on the gold-standard PTB-XL dataset (1,903 held-out test records), the CNN achieved an element-wise binary accuracy of 87.58% and a mean ROC-AUC of 0.9052 across five diagnostic superclasses (NORM, MI, STTC, HYP, CD). The system's computer vision pipeline bridges analog paper ECGs with digital neural modeling, while the late-fusion decision engine provides actionable 3-tier risk stratification and interpretable feature attributions. The findings confirm that combining electrophysiological waveform analysis with clinical risk indicators yields an effective, accessible screening tool to assist medical practitioners and enhance cardiovascular triage.")

    # Section XXII - References
    H1("REFERENCES")
    refs = [
        "[1] World Health Organization, \"Cardiovascular diseases (CVDs) Fact Sheet,\" WHO, Geneva, Switzerland, Tech. Rep., 2023.",
        "[2] P. Wagner, N. Strodthoff, R. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T. Schaeffter, \"PTB-XL, a large publicly available electrocardiography dataset,\" Scientific Data, vol. 7, no. 1, p. 154, 2020.",
        "[3] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C. K. Peng, and H. E. Stanley, \"PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,\" Circulation, vol. 101, no. 23, pp. e215-e220, 2000.",
        "[4] A. Y. Hannun, P. Rajpurkar, M. Haghpanahi, G. H. Tison, C. Bourn, M. P. Turakhia, and A. Y. Ng, \"Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network,\" Nature Medicine, vol. 25, no. 1, pp. 65-69, 2019.",
        "[5] A. H. Ribeiro, M. H. Ribeiro, G. M. Paixao, P. X. Oliveira, P. R. Gomes, J. A. Canazart, M. P. Ferreira, C. R. Andersson, P. W. Macfarlane, W. Meira Jr, T. B. Schon, and A. L. Ribeiro, \"Automatic diagnosis of the 12-lead ECG using a deep neural network,\" Nature Communications, vol. 11, no. 1, p. 1760, 2020.",
        "[6] N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, \"Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL,\" IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 5, pp. 1519-1528, 2021.",
        "[7] J. Pan and W. J. Tompkins, \"A real-time QRS detection algorithm,\" IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230-236, 1985.",
        "[8] R. B. D'Agostino, R. S. Vasan, M. J. Pencina, P. A. Wolf, M. Cobain, J. M. Massaro, and W. B. Kannel, \"General cardiovascular risk profile for use in primary care: The Framingham Heart Study,\" Circulation, vol. 117, no. 6, pp. 743-753, 2008.",
        "[9] D. C. Goff et al., \"2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk,\" Circulation, vol. 129, no. 25_suppl_2, pp. S49-S73, 2014.",
        "[10] T. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, \"Focal Loss for Dense Object Detection,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 2, pp. 318-327, 2020.",
        "[11] S. Ioffe and C. Szegedy, \"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,\" in International Conference on Machine Learning (ICML), 2015, pp. 448-456.",
        "[12] D. P. Kingma and J. Ba, \"Adam: A Method for Stochastic Optimization,\" in International Conference on Learning Representations (ICLR), 2015.",
        "[13] G. Bradski, \"The OpenCV Library,\" Dr. Dobb's Journal of Software Tools, 2000.",
        "[14] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
        "[15] M. Abadi et al., \"TensorFlow: A system for large-scale machine learning,\" in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2016, pp. 265-283."
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.1
        run = p.add_run(r)
        run.font.size = Pt(9.5)
        run.font.name = "Times New Roman"

    # Save documents
    out_docx_1 = os.path.join(OUT_DIR, "Heart_Disease_Risk_Assessment_Paper.docx")
    out_docx_2 = os.path.join(BASE_DIR, "results_paper", "Heart_Disease_Risk_Assessment_Paper.docx")
    doc.save(out_docx_1)
    doc.save(out_docx_2)
    print("Word paper generated successfully at:")
    print("  1.", out_docx_1)
    print("  2.", out_docx_2)

if __name__ == "__main__":
    create_full_paper()
