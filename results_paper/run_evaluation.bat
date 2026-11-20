@echo off
rem ============================================================
rem  Heart Disease Risk Assessment Agent
rem  Paper-Ready Results & Figures Generator  (ANALYSIS ONLY)
rem
rem  * Reads the existing project data / model / saved
rem    probabilities.  It never trains, re-splits, re-labels or
rem    overwrites anything inside your project.
rem  * All output is written to  results_paper\
rem
rem  Usage: double-click this file
rem ============================================================

cd /d "%~dp0.."

set PY=ecg_env\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Python environment not found: %PY%
    echo Expected the venv at  ecg_env\Scripts\python.exe  in the project root.
    echo Edit the PY path above if your environment lives elsewhere.
    pause
    exit /b 1
)

echo ============================================================
echo  Generating paper-ready results from the ACTUAL project data
echo  Data  : dataset\processed
echo  Model : models\ecg_cnn_improved_best.keras
echo  Probs : results\Y_test_probabilities.npy
echo  Output: results_paper\
echo ============================================================
echo.

"%PY%" results_paper\generate_paper_results.py

echo.
echo ------------------------------------------------------------
echo  Done.
echo  Figures : results_paper\figures
echo  Tables  : results_paper\tables
echo  Numbers : results_paper\tables\metrics_all.json
echo ------------------------------------------------------------
pause
