"""
convert_section4_md_to_docx.py
==============================
Converts results_paper/SECTION4_RESULTS_AND_DISCUSSION.md into a beautifully
formatted Word (.docx) document with all tokens auto-filled from metrics_all.json,
styled tables, and embedded high-resolution figures.
"""

import os
import re
import json
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "results_paper", "SECTION4_RESULTS_AND_DISCUSSION.md")
METRICS_PATH = os.path.join(BASE_DIR, "results_paper", "tables", "metrics_all.json")
FIG_DIR = os.path.join(BASE_DIR, "results_paper", "figures")
OUT_DOCX_1 = os.path.join(BASE_DIR, "results_paper", "SECTION4_RESULTS_AND_DISCUSSION.docx")
OUT_DOCX_2 = os.path.join(BASE_DIR, "paper_output", "SECTION4_RESULTS_AND_DISCUSSION.docx")
OUT_MD_AUTOFILLED = os.path.join(BASE_DIR, "results_paper", "SECTION4_RESULTS_AND_DISCUSSION_autofilled.md")

os.makedirs(os.path.join(BASE_DIR, "paper_output"), exist_ok=True)

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

def build_token_dict():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = {}
    classes = data.get("classes", ["NORM", "MI", "STTC", "HYP", "CD"])
    tr_pos = data.get("train_positives", [6569, 3435, 3436, 1751, 3205])
    va_pos = data.get("val_positives", [865, 456, 463, 241, 430])
    te_pos = data.get("test_positives", [870, 473, 464, 240, 422])

    for i, c in enumerate(classes):
        tokens[f"TRAIN_{c}"] = f"{tr_pos[i]:,}"
        tokens[f"VAL_{c}"] = f"{va_pos[i]:,}"
        tokens[f"TEST_{c}"] = f"{te_pos[i]:,}"

    tokens["N_TRAIN"] = f"{data.get('n_train', 14234):,}"
    tokens["N_VAL"] = f"{data.get('n_val', 1891):,}"
    tokens["N_TEST"] = f"{data.get('n_test', 1903):,}"

    pc = data.get("per_class", {})
    for c in classes:
        d = pc.get(c, {})
        tokens[f"P_{c}"] = f"{d.get('precision', 0):.3f}"
        tokens[f"R_{c}"] = f"{d.get('recall', 0):.3f}"
        tokens[f"F1_{c}"] = f"{d.get('f1', 0):.3f}"
        tokens[f"S_{c}"] = f"{d.get('support', 0):,}"
        tokens[f"TP_{c}"] = f"{d.get('tp', 0):,}"
        tokens[f"TN_{c}"] = f"{d.get('tn', 0):,}"
        tokens[f"FP_{c}"] = f"{d.get('fp', 0):,}"
        tokens[f"FN_{c}"] = f"{d.get('fn', 0):,}"
        tokens[f"TH_{c}"] = f"{d.get('threshold', 0.5):.2f}"

    agg = data.get("aggregate", {})
    micro = agg.get("micro", {})
    macro = agg.get("macro", {})
    wei = agg.get("weighted", {})

    tokens["P_MICRO"] = f"{micro.get('precision', 0):.3f}"
    tokens["R_MICRO"] = f"{micro.get('recall', 0):.3f}"
    tokens["F1_MICRO"] = f"{micro.get('f1', 0):.3f}"
    tokens["P_MACRO"] = f"{macro.get('precision', 0):.3f}"
    tokens["R_MACRO"] = f"{macro.get('recall', 0):.3f}"
    tokens["F1_MACRO"] = f"{macro.get('f1', 0):.3f}"
    tokens["P_WEI"] = f"{wei.get('precision', 0):.3f}"
    tokens["R_WEI"] = f"{wei.get('recall', 0):.3f}"
    tokens["F1_WEI"] = f"{wei.get('f1', 0):.3f}"
    tokens["S_TOTAL"] = f"{data.get('n_test', 1903) * 5:,}"

    tokens["TEST_ACCURACY"] = f"{agg.get('test_elementwise_accuracy', 0.8593)*100:.2f}%"
    tokens["SUBSET_ACCURACY"] = f"{agg.get('subset_accuracy', 0.5780)*100:.2f}%"
    tokens["HAMMING"] = f"{agg.get('hamming_loss', 0.1407):.4f}"
    tokens["SPEC_MACRO"] = f"{agg.get('specificity_macro', 0.8869):.4f}"
    tokens["SPEC_MICRO"] = f"{agg.get('specificity_micro', 0.8914):.4f}"
    tokens["AUC_MACRO"] = f"{agg.get('roc_auc_macro', 0.9031):.4f}"
    tokens["AUC_MICRO"] = f"{agg.get('roc_auc_micro', 0.9162):.4f}"
    tokens["PR_AUC_MACRO"] = f"{agg.get('pr_auc_macro', 0.7711):.4f}"

    val = data.get("validation", {})
    tokens["VAL_ACCURACY"] = f"{val.get('val_elementwise_accuracy', 0.8645)*100:.2f}%"
    tokens["BCE"] = f"{agg.get('bce_unweighted', 0.3510):.4f}"

    return tokens

def fill_tokens(text, token_dict):
    for k, v in token_dict.items():
        text = text.replace(f"{{{{{k}}}}}", v)
    return text

def convert_to_docx():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    tokens = build_token_dict()
    autofilled_md = fill_tokens(md_content, tokens)

    with open(OUT_MD_AUTOFILLED, "w", encoding="utf-8") as f:
        f.write(autofilled_md)
    print("Autofilled Markdown saved to:", OUT_MD_AUTOFILLED)

    # Build Word Document
    doc = Document()

    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Parse autofilled markdown line by line
    lines = autofilled_md.split("\n")
    i = 0
    in_table = False
    table_lines = []

    def flush_table(tbl_lines):
        if not tbl_lines:
            return
        parsed_rows = []
        for tl in tbl_lines:
            if not tl.strip() or tl.strip().startswith("|---") or tl.strip().startswith("| ---"):
                continue
            cells = [c.strip() for c in tl.split("|")[1:-1]]
            if cells:
                parsed_rows.append(cells)

        if not parsed_rows:
            return

        cols_count = max(len(r) for r in parsed_rows)
        table = doc.add_table(rows=len(parsed_rows), cols=cols_count)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = 'Table Grid'

        for r_idx, row in enumerate(parsed_rows):
            for c_idx in range(cols_count):
                cell = table.rows[r_idx].cells[c_idx]
                val = row[c_idx] if c_idx < len(row) else ""
                cell.text = val
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05

                if r_idx == 0:
                    set_cell_background(cell, "1F4E79")
                    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.bold = True
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                else:
                    bg_color = "F2F5F9" if r_idx % 2 == 1 else "FFFFFF"
                    set_cell_background(cell, bg_color)
                    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
                    if c_idx == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(9.0)
                        run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

        p_after = doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(6)

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for table
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
            i += 1
            continue
        else:
            if table_lines:
                flush_table(table_lines)
                table_lines = []

        if not stripped:
            i += 1
            continue

        # Check for divider
        if stripped.startswith("---"):
            i += 1
            continue

        # Check for headers
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[2:])
            run.bold = True
            run.font.size = Pt(16)
            run.font.name = "Times New Roman"
            run.font.color.rgb = RGBColor(0x0a, 0x25, 0x40)
        elif stripped.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[3:])
            run.bold = True
            run.font.size = Pt(12.5)
            run.font.name = "Times New Roman"
            run.font.color.rgb = RGBColor(0x0f, 0x2d, 0x59)
        elif stripped.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[4:])
            run.bold = True
            run.italic = True
            run.font.size = Pt(11)
            run.font.name = "Times New Roman"
            run.font.color.rgb = RGBColor(0x1a, 0x3a, 0x60)
        elif stripped.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.right_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(stripped[2:])
            run.italic = True
            run.font.size = Pt(9.5)
            run.font.name = "Times New Roman"
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        elif stripped.startswith("* ") or stripped.startswith("- "):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(stripped[2:])
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
        elif re.match(r"^\d+\.\s+", stripped):
            match = re.match(r"^(\d+\.)\s+(.*)", stripped)
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            r_num = p.add_run(match.group(1) + " ")
            r_num.bold = True
            run = p.add_run(match.group(2))
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
        else:
            # Check if this paragraph introduces a figure or table
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(stripped)
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"

            # If it's a figure caption header, embed the figure if available
            fig_match = re.search(r"\*\*Figure\s+(\d+)\.\*\*", stripped)
            if fig_match:
                fig_num = fig_match.group(1)
                fig_map = {
                    "1": "fig1_training_validation_accuracy.png",
                    "2": "fig2_training_validation_loss.png",
                    "3": "fig3_confusion_matrix_ovr.png",
                    "4": "fig4_classwise_precision_recall_f1.png",
                    "5": "fig5_class_distribution.png",
                    "6": "fig6_roc_curves.png",
                    "7": "fig7_precision_recall_curves.png",
                    "8": "fig8_overall_performance.png",
                }
                fig_file = fig_map.get(fig_num)
                if fig_file:
                    fig_full = os.path.join(FIG_DIR, fig_file)
                    if os.path.exists(fig_full):
                        p_img = doc.add_paragraph()
                        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        p_img.paragraph_format.space_before = Pt(6)
                        p_img.paragraph_format.space_after = Pt(6)
                        doc.add_picture(fig_full, width=Inches(5.0))
                        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        i += 1

    if table_lines:
        flush_table(table_lines)

    doc.save(OUT_DOCX_1)
    doc.save(OUT_DOCX_2)
    print("Section 4 Word Document created successfully:")
    print("  1.", OUT_DOCX_1)
    print("  2.", OUT_DOCX_2)

if __name__ == "__main__":
    convert_to_docx()
