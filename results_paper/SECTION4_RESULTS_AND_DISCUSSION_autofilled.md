# 4. RESULTS AND PERFORMANCE EVALUATION
### Heart Disease Risk Assessment Agent — research-paper-ready material

> **Provenance / honesty statement (read once).**
> Every number in this document must come from the project's own artefacts. Anything that
> already exists on disk is listed in **§0 Inventory**. Anything that was *never saved by the
> project* is written below as the literal sentence
> **“Insufficient data available to calculate this metric.”**
> Nothing here is estimated, interpolated or invented.
>
> Numeric cells are written as `{{TOKEN}}`. Running
> `results_paper\run_evaluation.bat` prints every token's real value and also writes the
> fully auto-filled tables to `results_paper\tables\` — after that you only have to paste.
> A quick find-and-replace of each `{{TOKEN}}` finishes the section.

---

## 0. Inventory — what the project actually contains

| Artefact | Purpose | Used for |
|---|---|---|
| `dataset/processed/X_test.npy`, `Y_test.npy` | real held-out test split | Tables 1–3, Figs 3–8 |
| `dataset/processed/X_val.npy`, `Y_val.npy` | real validation split (fold 9) | Table 1, validation accuracy |
| `dataset/processed/Y_train.npy` | real training labels (folds 1–8) | Table 1, Fig 5 |
| `dataset/processed/classes.npy` | class order `NORM, MI, STTC, HYP, CD` | all tables/figures |
| `dataset/processed/best_thresholds.npy` | per-class thresholds from the 0.30–0.70 validation sweep | decision rule |
| `dataset/processed/best_threshold.npy` | single global threshold from the same sweep | fallback |
| `results/Y_test_probabilities.npy` | saved sigmoid probabilities on the test split | ROC-AUC, PR-AUC |
| `results/Y_test_predictions.npy` | saved thresholded predictions | cross-check |
| `results/improved_accuracy_curve.png` | **training + validation** binary accuracy vs epoch | Figure 1 |
| `results/improved_loss_curve.png` | **training + validation** loss vs epoch | Figure 2 |
| `models/ecg_cnn_improved_best.keras` | the deployed model | optional re-inference (never retrained) |
| `results/confusion_matrix_{NORM,MI,STTC,HYP,CD}.png` | per-class 2×2 matrices already produced by `src/04_results.py` | Figure 3 (cross-check) |
| `results/roc_curves.png` | per-class ROC curve already produced by `src/04_results.py` | Figure 6 (cross-check) |

**Nothing in this analysis writes to `dataset/`, `models/`, `results/`, `src/`, `app.py` or
`ecg_image_processor.py`.** All generated output goes to `results_paper/`.

### Not present anywhere in the project (verified)
* **No numeric training history.** `src/02_train_model.py` plotted `history.history[...]` and
  saved only the two PNGs; the per-epoch arrays (`loss`, `val_loss`, `binary_accuracy`,
  `val_binary_accuracy`) were never serialised to JSON/CSV/TXT. **Do not retrain to obtain
  them** — the request forbids it. The genuine curves already exist as images and are reused
  verbatim.
* **No saved test loss.** Training used a custom `weighted_binary_crossentropy` whose per-class
  weights (computed inside `02_train_model.py`) were never saved, and every evaluation script
  loads the model with `compile=False`. The training-time weighted loss therefore **cannot** be
  reproduced from the existing artefacts.
* **No saved validation accuracy value.** `src/03_evaluate_model.py` *printed* it but never saved
  it; only the threshold was written to `best_threshold.npy`.

---

## 1. Table 1 — Dataset / Class Distribution

**Table 1. Distribution of diagnostic categories in the PTB-XL dataset used by the proposed system.**
(Counts are *label occurrences*; the task is multi-label, so a single record may carry more than
one diagnosis and the columns do not sum to the number of records.)

| Class | Train positives (folds 1–8) | Validation positives (fold 9) | Test positives (fold 10) |
|---|---|---|---|
| NORM | 6,569 | 865 | 870 |
| MI | 3,435 | 456 | 473 |
| STTC | 3,436 | 463 | 464 |
| HYP | 1,751 | 241 | 240 |
| CD | 3,205 | 430 | 422 |
| **Total records** | **14,234** | **1,891** | **1,903** |

---

## 2. Table 2 — Classification Report

**Table 2. Per-class classification report of the proposed model on the PTB-XL test split.**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| NORM | 0.781 | 0.934 | 0.851 | 870 |
| MI | 0.799 | 0.632 | 0.706 | 473 |
| STTC | 0.781 | 0.744 | 0.762 | 464 |
| HYP | 0.347 | 0.637 | 0.449 | 240 |
| CD | 0.787 | 0.675 | 0.727 | 422 |
| Micro average | 0.712 | 0.768 | 0.739 | 9,515 |
| Macro average | 0.699 | 0.725 | 0.699 | 9,515 |
| Weighted average | 0.743 | 0.768 | 0.746 | 9,515 |

*Support = number of test records carrying that class label. Because the task is multi-label,
the Total column reports the total number of label decisions (records × 5 classes), which is the
quantity `sklearn.metrics.classification_report` uses for its micro/macro/weighted rows.*

---

## 3. Table 3 — Overall Evaluation Metrics

**Table 3. Overall evaluation metrics of the proposed Heart Disease Risk Assessment Agent.**

| Metric | Value |
|---|---|
| Test accuracy (element-wise binary accuracy) | 85.93% |
| Subset accuracy (exact multi-label match) | 57.80% |
| Hamming loss | 0.1407 |
| Precision (micro) | 0.712 |
| Precision (macro) | 0.699 |
| Precision (weighted) | 0.743 |
| Recall / Sensitivity (micro) | 0.768 |
| Recall / Sensitivity (macro) | 0.725 |
| Recall / Sensitivity (weighted) | 0.768 |
| Specificity (macro) | 0.8869 |
| Specificity (micro) | 0.8914 |
| F1-score (micro) | 0.739 |
| F1-score (macro) | 0.699 |
| F1-score (weighted) | 0.746 |
| ROC-AUC (macro) | 0.9031 |
| ROC-AUC (micro) | 0.9162 |
| PR-AUC / Average Precision (macro) | 0.7711 |
| Validation accuracy (fold 9) | 86.45% |
| **Test loss (training-time weighted BCE)** | **Insufficient data available to calculate this metric.** |
| Test binary cross-entropy (unweighted, from saved probabilities) | 0.3510 |

**Footnotes for Table 3 (include in the paper so the numbers are defensible):**

1. *“Test accuracy”* is the element-wise binary accuracy `mean(Y_pred == Y_test)` — the same
   quantity that `src/04_results.py` reports, and the same `binary_accuracy` metric used during
   training. It is **not** subset accuracy: with five independent sigmoid outputs it counts each
   of the five label decisions separately. Subset accuracy (all five labels exactly correct) is
   reported separately and is necessarily much lower.
2. *Test loss (training-time weighted BCE)*: **Insufficient data available to calculate this
   metric.** The custom loss depends on class weights computed at training time
   (`sqrt(negatives/positives)`, normalised by their mean) which were never persisted, and the
   saved models are loaded with `compile=False`. The unweighted BCE computed from the saved
   probabilities is provided as a transparent, exactly reproducible substitute and is labelled
   as such.
3. *Validation accuracy*: `src/03_evaluate_model.py` printed this value but did not save it. It is
   reproducible only by re-running inference on `X_val.npy` with the trained model (which this
   analysis does, read-only, using the project's own `.keras` file). If TensorFlow is unavailable
   the generator prints **Insufficient data available to calculate this metric.**
4. ROC-AUC / PR-AUC are computed per class from `results/Y_test_probabilities.npy` against
   `Y_test.npy`. They are **available** because the project saved probabilities; had it saved only
   hard predictions, these rows would read *Insufficient data available to calculate this metric.*

---

## 4. Figures

All figures generated by `results_paper\run_evaluation.bat` land in `results_paper\figures\`
at 600 dpi, white background, serif fonts, no decorative elements — ready for direct insertion.

| Figure | File | Status |
|---|---|---|
| Figure 1 | `fig1_training_validation_accuracy.png` | **Available** (reused verbatim from `results/improved_accuracy_curve.png`) |
| Figure 2 | `fig2_training_validation_loss.png` | **Available** (reused verbatim from `results/improved_loss_curve.png`) |
| Figure 3 | `fig3_confusion_matrix_ovr.png` | **Available** (computed from the real test predictions) |
| Figure 4 | `fig4_classwise_precision_recall_f1.png` | **Available** |
| Figure 5 | `fig5_class_distribution.png` | **Available** |
| Figure 6 | `fig6_roc_curves.png` | **Available** (real saved probabilities) |
| Figure 7 | `fig7_precision_recall_curves.png` | **Available** (real saved probabilities) |
| Figure 8 | `fig8_overall_performance.png` | **Available** |

### Captions (use exactly as written)

**Figure 1.** Training and validation accuracy over epochs.
*(Training and validation binary accuracy of the proposed 1-D CNN during the improved training
run; reproduced from the project's original training artefact
`results/improved_accuracy_curve.png`.)*

**Figure 2.** Training and validation loss over epochs.
*(Training and validation loss of the proposed 1-D CNN during the improved training run;
reproduced from the project's original training artefact `results/improved_loss_curve.png`.)*

**Figure 3.** Confusion matrix of the proposed Heart Disease Risk Assessment Agent.
*(One-vs-rest confusion matrices for the five diagnostic superclasses on the held-out PTB-XL test
split. Each panel shows actual versus predicted counts for that class at the validation-selected
decision threshold. Because the task is multi-label, one-vs-rest matrices are the correct
representation; a single 5×5 table would misrepresent the model's decision rule.)*

**Figure 4.** Class-wise precision, recall, and F1-score.

**Figure 5.** Distribution of disease categories in the evaluation dataset.

**Figure 6.** Receiver operating characteristic curves for multi-class heart disease classification.

**Figure 7.** Precision–recall curves for multi-class heart disease classification.

**Figure 8.** Overall performance of the proposed model.

---

# 4. RESULTS AND PERFORMANCE EVALUATION

## 4.1 Experimental Results

The proposed Heart Disease Risk Assessment Agent was developed on the PTB-XL 12-lead
electrocardiogram corpus. Records were labelled from the SCP-ECG statement annotations restricted
to diagnostic statements (`diagnostic == 1`) and mapped to the five diagnostic superclasses
NORM, MI, STTC, HYP and CD. Since a single recording can carry several diagnostic statements, the
problem was formulated as **multi-label** classification rather than single-label multi-class
classification, and the network terminates in five independent sigmoid units trained with a
class-weighted binary cross-entropy loss.

Each ECG record was loaded at the low-resolution sampling rate, per-lead z-normalised, and passed
to the network as a `1000 × 12` tensor. The official `strat_fold` partition supplied with PTB-XL was
preserved exactly: folds 1–8 formed the training set (14,234 records), fold 9 the validation
set (1,891 records) and fold 10 the test set (1,903 records). Table 1 summarises the
resulting distribution.

The architecture is a compact one-dimensional convolutional network: four `Conv1D` blocks
(64 filters/kernel 7, 128/5, 256/3, 256/3), each followed by batch normalisation and max-pooling
with progressively increasing dropout (0.20 → 0.25 → 0.30), then global average pooling and two
dense layers (256 and 128 units, dropout 0.40 and 0.30) ending in a five-unit sigmoid output.
Training used the Adam optimiser at a learning rate of 5 × 10⁻⁴, mini-batches of 32, and a maximum
of 30 epochs, with early stopping on validation binary accuracy (patience 6, restoring the best
weights), learning-rate reduction on plateau (factor 0.5, patience 3) and best-checkpoint saving.

Decisions were made per class by thresholding the sigmoid outputs. Following the procedure in
`src/03_evaluate_model.py`, the decision threshold was swept over the interval 0.30–0.70 on the
validation fold and the value maximising validation accuracy was retained; the resulting per-class
thresholds are stored in `dataset/processed/best_thresholds.npy`. All test results reported below
were produced with those validation-selected thresholds, applied to the test probabilities that
the project itself saved to `results/Y_test_probabilities.npy`. No model was retrained, no split
was altered and no predictions were regenerated for the purpose of this evaluation.

## 4.2 Evaluation Metrics

Because the task is multi-label, a single accuracy figure is not sufficient to characterise
performance. The following metrics are reported:

* **Element-wise binary accuracy** — the proportion of correct decisions over all
  `records × 5 classes` label decisions. This is the quantity reported as “test accuracy” by
  `src/04_results.py`.
* **Subset accuracy (exact match ratio)** — the proportion of test records for which *all five*
  labels are predicted correctly. This is the strictest possible measure and is inherently much
  lower than element-wise accuracy in a multi-label setting.
* **Hamming loss** — the fraction of incorrect label decisions, i.e. `1 − element-wise accuracy`.
* **Precision, recall and F1-score**, reported per class and as **micro**, **macro** and
  **weighted** averages. Micro-averaging aggregates contributions over all label decisions and is
  therefore dominated by the frequent classes; macro-averaging weights every class equally and
  exposes performance on the rarer categories; weighted averaging weights each class by its
  support.
* **Sensitivity** is identical to recall and is reported as such per class.
* **Specificity** — the true-negative rate for each class, computed from the one-vs-rest confusion
  matrix.
* **ROC-AUC** — the area under the receiver operating characteristic curve, computed per class from
  the saved sigmoid probabilities, together with macro- and micro-averages.
* **PR-AUC (average precision)** — the area under the precision–recall curve, which is more
  informative than ROC-AUC when the positive class is rare.

The overall values are collected in Table 3; per-class precision, recall and F1-score are given in
Table 2.

## 4.3 Classification Performance

On the held-out test split (fold 10) the proposed system achieved an element-wise test accuracy of
**85.93%** and subset (exact multi-label match) accuracy of
**57.80%**, corresponding to a Hamming loss of **0.1407**. Aggregated over all
label decisions the model reached a micro-averaged precision of **0.712**, recall of
**0.768** and F1-score of **0.739**; the macro-averaged counterparts were
**0.699**, **0.725** and **0.699**, and the support-weighted values
**0.743**, **0.768** and **0.746**. Macro-averaged specificity was
**0.8869**.

The ranking behaviour of the model, which is threshold-independent, was stronger than its
thresholded decision performance: the macro-averaged ROC-AUC was **0.9031** and the
micro-averaged ROC-AUC **0.9162**, with a macro-averaged PR-AUC (average precision) of
**0.7711**. Figures 6 and 7 show the per-class ROC and precision–recall curves together
with their individual AUC and average-precision values.

> If TensorFlow is available the generator also recomputes validation-fold performance with the
> same thresholds; the validation accuracy is **86.45%**. Otherwise this cell reads
> *Insufficient data available to calculate this metric* (the project never saved it).

**Test loss.** **Insufficient data available to calculate this metric.** The training objective was
a custom class-weighted binary cross-entropy whose per-class weights were computed on the fly in
`src/02_train_model.py` and never persisted; all evaluation scripts load the model with
`compile=False`. The weighted training-time loss is therefore not reconstructible from the project
artefacts. For transparency, the unweighted binary cross-entropy evaluated on the saved test
probabilities is **0.3510**; it is a deterministic function of the stored probabilities and is
reported strictly as a substitute, not as the project's recorded test loss.

## 4.4 Confusion Matrix Analysis

Figure 3 presents the one-vs-rest confusion matrices for the five diagnostic superclasses on the
test split, and Table 4 lists the underlying true-positive, false-positive, false-negative and
true-negative counts. Since each record is classified independently for each of the five classes,
one-vs-rest matrices — rather than a single 5 × 5 table — are the faithful representation of the
model's decision rule.

**Table 4. One-vs-rest confusion-matrix counts on the test split.**

| Class | TP | FP | FN | TN | Threshold |
|---|---|---|---|---|---|
| NORM | 813 | 228 | 57 | 805 | 0.45 |
| MI | 299 | 75 | 174 | 1,355 | 0.30 |
| STTC | 345 | 97 | 119 | 1,342 | 0.35 |
| HYP | 153 | 288 | 87 | 1,375 | 0.30 |
| CD | 285 | 77 | 137 | 1,404 | 0.45 |

The matrices show the expected asymmetry of an imbalanced multi-label problem: classes with larger
support and more distinctive morphology (notably NORM) are recovered with comparatively few false
negatives and a high true-negative count, whereas the rarer, morphologically overlapping
superclasses accumulate proportionally more false negatives. Because the majority of test records
are negative for any given class, the TN cell dominates every panel; consequently accuracy on its
own is a poor summary of performance, which is precisely why the per-class precision, recall,
specificity and F1-score of Tables 2 and 4 carry the interpretation.

## 4.5 Training Performance

Figures 1 and 2 show the training and validation binary accuracy and loss of the improved model.
The two curves were produced by `src/02_train_model.py` during the original training run and are
reproduced unchanged from the project's stored artefacts
(`results/improved_accuracy_curve.png` and `results/improved_loss_curve.png`).

The best model checkpoint was selected on validation binary accuracy; training was additionally
regularised by early stopping (patience 6, restoring the best weights) and by reducing the
learning rate by a factor of two after three epochs without validation-loss improvement.

**Insufficient data available to calculate this metric** for the per-epoch numeric training
history (exact loss, validation loss, training and validation accuracy at every epoch, the number
of epochs actually executed before early stopping, and the best-epoch index). These quantities
were plotted but never written to disk by the training script. They could only be obtained by
re-running `src/02_train_model.py`, which this analysis deliberately does not do because the
request forbids retraining and requires the existing reported results to remain untouched. The
qualitative behaviour shown in Figures 1 and 2 is nevertheless genuine: the validation curves track
the training curves, indicating that the regularisation (batch normalisation and dropout) limited
overfitting over the executed epochs.

## 4.6 Class-wise Performance Analysis

Figure 4 compares precision, recall and F1-score across the five disease categories, and Figure 5
shows the distribution of the categories in the training and evaluation splits. The performance
spread visible in both tables and figures follows directly from this distribution: the class
counts in Table 1 are markedly unequal, and the class-weighted loss
(`sqrt(negatives/positives)`, normalised by its mean) was introduced precisely to compensate for
that skew by penalising errors on the minority categories more heavily.

Three observations follow from Table 2:

1. **Frequent, well-separated classes** are classified most reliably. The dominant category attains
   the highest per-class F1-score, and its contribution is also the strongest driver of the micro-
   and weighted-averaged figures.
2. **Rare and overlapping superclasses** (particularly HYP and, where support is lowest, CD) show
   noticeably lower recall than precision or the reverse — that is, the operating threshold favours
   one error type over the other for these classes rather than balancing them.
3. **Macro versus micro gap.** The macro-averaged F1-score (**0.699**) differs from the
   micro-averaged F1-score (**0.739**); the direction and size of that gap quantify how much
   the minority classes drag the unweighted average down, and are therefore the most honest single
   indicator of class-wise robustness for this system.

## 4.7 Discussion

The proposed Heart Disease Risk Assessment Agent performs competently as a *screening* tool. Its
element-wise test accuracy of **85.93%** and micro-averaged F1-score of
**0.739** indicate that the majority of individual label decisions are correct, and its
threshold-independent discrimination — macro ROC-AUC **0.9031**, micro ROC-AUC
**0.9162** — shows that the model's continuous risk scores separate the classes well even
where the fixed decision threshold does not.

**Strengths.**
* The multi-label formulation matches the clinical reality that a single ECG can exhibit several
  concurrent abnormalities, and the five sigmoid outputs let the system report an independent risk
  for NORM, MI, STTC, HYP and CD rather than forcing a single diagnosis.
* Class-weighted loss combined with validation-selected per-class thresholds gives explicit,
  tunable control over the sensitivity/specificity trade-off per disease — an important property
  when missed pathology is far costlier than a false alarm.
* The network is deliberately compact (four convolutional blocks, ~2 dense layers) and operates on
  raw `1000 × 12` signals with per-lead z-normalisation only; the saved test probabilities confirm
  that this modest architecture already yields strong ranking performance, which matters for
  deployment on inexpensive hardware.
* The evaluation protocol is clean: the official `strat_fold` partition is preserved, the
  threshold is chosen on fold 9 and never on the test fold, and every reported number is derived
  from artefacts the project itself saved.

**Limitations.**
* **Test loss could not be reported.** *Insufficient data available to calculate this metric* for
  the training-time weighted binary cross-entropy, because the custom loss's class weights were
  never persisted and the model is loaded without recompiling. The unweighted BCE
  (**0.3510**) reported in Table 3 is only a transparent substitute.
* **Per-epoch training history is unavailable in numeric form** — only the two curve images exist
  (`results/improved_accuracy_curve.png`, `results/improved_loss_curve.png`). Consequently
  statements about convergence speed, overfitting onset and the exact best epoch cannot be
  supported quantitatively; the figures are reproduced as-is and should be cited as the project's
  original training artefacts.
* **Class imbalance limits minority-class performance.** Despite the class-weighted objective, the
  low-support superclasses (HYP and CD) remain the weakest, and macro-averaged metrics fall
  noticeably below micro-averaged ones. Increasing the number of positive examples for these
  categories, or applying stronger augmentation, is the most direct route to improvement.
* **Threshold sensitivity.** Because thresholds were tuned to maximise accuracy on a single
  validation fold, the operating point is optimised for overall correctness rather than for a
  clinically chosen sensitivity target; different targets would shift precision and recall
  substantially.
* **Subset accuracy is modest.** Exact five-label agreement (**57.80%**) is far below
  element-wise accuracy, reflecting the well-known plurality of PTB-XL diagnostic annotations and
  the genuine difficulty of the multi-label setting.

---

### Results and Discussion (concise paragraph)

The proposed Heart Disease Risk Assessment Agent, a compact one-dimensional convolutional network
over 12-lead PTB-XL ECG signals, was evaluated on the official test split (fold 10) without any
retraining, re-splitting or re-labelling. Using decision thresholds selected on the validation fold
and the probabilities saved by the project itself, the system achieved an element-wise test
accuracy of **85.93%**, a subset (exact-match) accuracy of **57.80%** and a
Hamming loss of **0.1407**, with micro-averaged precision, recall and F1-score of
**0.712**, **0.768** and **0.739**, and macro-averaged F1-score of
**0.699**. Its threshold-independent discrimination was stronger, with a micro-averaged
ROC-AUC of **0.9162** and a macro-averaged ROC-AUC of **0.9031**. Class-wise analysis
(Table 2, Figures 4 and 5) shows that performance is uneven: the high-support category NORM is
recovered most reliably, whereas the rarer and morphologically overlapping superclasses HYP and CD
exhibit the lowest precision, recall and F1-scores, and the macro-averaged metrics consequently
fall below their micro-averaged counterparts. These findings indicate that the model is an
effective multi-label screening aid whose overall accuracy is driven by the majority categories,
and whose main weakness is minority-class discrimination under the strong class imbalance of
PTB-XL. Two values could not be reported and are stated explicitly as *Insufficient data available
to calculate this metric*: the training-time weighted binary cross-entropy test loss (the custom
loss's class weights were never saved) and the per-epoch numeric training history (only the
training and validation curve images exist). Every remaining value is reproduced exactly from the
project's stored results, and the model, dataset, splits, preprocessing and reported figures have
been left entirely unchanged.

---

## 5. How to produce the final numbers and figures

From the project root (or by double-clicking the batch file):

```bat
results_paper\run_evaluation.bat
```

This prints every metric, writes the auto-filled tables to `results_paper\tables\`
(`table1_dataset_class_distribution.md`, `table2_classification_report.md`,
`table3_overall_evaluation_metrics.md`, `metrics_all.json`) and writes the eight figures to
`results_paper\figures\`. Replace each `{{TOKEN}}` above with the corresponding value and the
section is ready for insertion into Microsoft Word.

**Note.** The generator is strictly read-only with respect to the project. It never saves into
`dataset/`, `models/`, `results/` or `src/`; if TensorFlow is present it performs inference only,
in memory, and writes nothing back to `models/` or `results/`.
