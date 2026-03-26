# 🧠 MindCheck — Mental Disorder Diagnosis ML

**University of Eastern Africa, Baraton, Eldoret, Kenya**  
*Isaiah Gidayi Aduma · Ian Betto · Ezra Bolo*

---

## Overview

A machine-learning system that screens for symptoms of **Depression, Anxiety and Stress** using the DASS-21 (Depression Anxiety Stress Scales) questionnaire.

The model uses a **Random Forest classifier** (tuned via GridSearchCV with 5-fold cross-validation) to classify responses as indicating a potential mental health concern or not.

A full **Python desktop application** (Tkinter) allows users to complete the DASS-21 questionnaire and receive an instant screening result.

---

## Project Structure

```
mental_disorder_project/
│
├── generate_dataset.py     # Step 1 – Generate DASS-21 dataset
├── train_model.py          # Step 2 – Train, tune and evaluate ML models
├── app.py                  # Step 3 – Desktop GUI application
│
├── data/
│   └── mental_disorders.csv        # Generated dataset
│
├── models/
│   ├── best_model.pkl              # Trained Random Forest
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoder.pkl           # Gender encoder
│   └── feature_names.pkl           # Feature list
│
└── reports/
    ├── model_comparison.csv        # Performance table
    ├── classification_report.txt   # Precision / Recall / F1 per class
    ├── confusion_matrix.png        # Confusion matrix heatmap
    ├── model_comparison.png        # Bar chart: 3 models × 4 metrics
    ├── feature_importance.png      # Top-15 feature importances
    └── crossval_boxplot.png        # 5-fold CV F1 distributions
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

> Requires Python 3.8+. Tkinter is included in standard Python installations.

### 2. Generate the dataset

```bash
python generate_dataset.py
```

This creates `data/mental_disorders.csv` with 2,000 realistic DASS-21 samples including demographic features. If you have a real dataset, place it at this path with a `target` column (0 = no disorder, 1 = disorder).

### 3. Train the model

```bash
python train_model.py
```

This will:
- Run 5-fold stratified cross-validation on 3 classifiers
- Tune the Random Forest with GridSearchCV
- Print a full performance comparison table
- Save the best model to `models/`
- Save 4 evaluation plots to `reports/`

### 4. Launch the app

```bash
python app.py
```

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **~92%** | **~90%** | **~88%** | **~89%** |
| Support Vector Machine | ~89% | ~87% | ~85% | ~86% |
| Logistic Regression | ~84% | ~82% | ~80% | ~81% |

*Exact values depend on the dataset used. Run `train_model.py` to see your results.*

---

## DASS-21 Severity Thresholds

| Subscale | Normal | Mild | Moderate | Severe | Extremely Severe |
|---|---|---|---|---|---|
| Depression | 0–4 | 5–6 | 7–10 | 11–13 | 14+ |
| Anxiety | 0–3 | 4–5 | 6–7 | 8–9 | 10+ |
| Stress | 0–7 | 8–9 | 10–12 | 13–16 | 17+ |

---

## GitHub

https://github.com/Omurwa01/mentaldisorderAI.git

---

## Disclaimer

> This tool is an academic project and is **NOT** a substitute for professional mental health diagnosis or treatment. If you are experiencing symptoms of depression, anxiety or stress, please consult a qualified mental health professional.
