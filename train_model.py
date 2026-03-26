"""
train_model.py
--------------
Improved ML pipeline for Mental Disorder Diagnosis.
  - Loads and preprocesses the DASS-21 dataset
  - Encodes categoricals, scales features
  - Trains 3 classifiers
  - 5-Fold Stratified Cross-Validation
  - GridSearchCV hyperparameter tuning for Random Forest
  - Full evaluation: accuracy, precision, recall, F1, confusion matrix
  - Saves best model + scaler + feature list
  - Generates & saves evaluation plots
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

DATASET_PATH = "data/mental_disorders.csv"

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    print("Dataset not found. Run generate_dataset.py first.")
    raise FileNotFoundError(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)
print(f"Loaded dataset: {df.shape}  |  Class dist: {df['target'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

# Encode gender
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# Store feature names BEFORE scaling
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / test split (80 / 20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Cross-Validation (baseline)
# ─────────────────────────────────────────────
print("\n── 5-Fold Cross-Validation ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_models = {
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine" : SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
}

cv_results = {}
for name, model in baseline_models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    cv_results[name] = scores
    print(f"  {name:30s}  F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. Hyperparameter Tuning (Random Forest)
# ─────────────────────────────────────────────
print("\n── Tuning Random Forest ──")
param_grid = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
print(f"  Best params : {grid_search.best_params_}")
print(f"  Best CV F1  : {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# 5. Train All Final Models
# ─────────────────────────────────────────────
final_models = {
    "Random Forest"          : grid_search.best_estimator_,
    "Support Vector Machine" : SVC(kernel='rbf', probability=True, random_state=42),
    "Logistic Regression"    : LogisticRegression(max_iter=1000, random_state=42),
}

trained = {}
for name, model in final_models.items():
    model.fit(X_train, y_train)
    trained[name] = model

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
print("\n── Test-Set Evaluation ──")
results = []
predictions = {}

for name, model in trained.items():
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    row = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred), 4),
        "F1-Score" : round(f1_score(y_test, y_pred), 4),
    }
    results.append(row)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("reports/model_comparison.csv", index=False)

# Best model
best_idx        = results_df["F1-Score"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
best_model      = trained[best_model_name]
print(f"\n★  Best model: {best_model_name}")

# Classification report
y_best_pred = predictions[best_model_name]
report_str = classification_report(y_test, y_best_pred,
                                   target_names=["No Disorder", "Disorder"])
print("\n── Classification Report ──")
print(report_str)
with open("reports/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n\n")
    f.write(report_str)

# ─────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────

# 7a. Confusion Matrix
cm = confusion_matrix(y_test, y_best_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disorder", "Disorder"],
            yticklabels=["No Disorder", "Disorder"], ax=ax)
ax.set_title(f"Confusion Matrix – {best_model_name}", fontsize=13, fontweight='bold')
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
plt.close()
print("Saved → reports/confusion_matrix.png")

# 7b. Model Comparison Bar Chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A868']

fig, ax = plt.subplots(figsize=(9, 5))
for i, row in results_df.iterrows():
    ax.bar(x + i * width, [row[m] for m in metrics],
           width, label=row["Model"], color=colors[i], alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=150)
plt.close()
print("Saved → reports/model_comparison.png")

# 7c. Feature Importance (Random Forest only)
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, y='feature', x='importance', palette='viridis', ax=ax)
    ax.set_title("Top 15 Feature Importances – Random Forest",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    plt.close()
    print("Saved → reports/feature_importance.png")

# 7d. Cross-Val F1 Box Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(cv_results.values(), labels=cv_results.keys(), patch_artist=True,
           boxprops=dict(facecolor='#4C72B0', alpha=0.7))
ax.set_title("5-Fold Cross-Validation F1 Scores", fontsize=13, fontweight='bold')
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.05)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("reports/crossval_boxplot.png", dpi=150)
plt.close()
print("Saved → reports/crossval_boxplot.png")

# ─────────────────────────────────────────────
# 8. Save Best Model + Scaler
# ─────────────────────────────────────────────
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(le,         "models/label_encoder.pkl")
print(f"\nSaved → models/best_model.pkl  (scaler.pkl, label_encoder.pkl)")
print("\n✓ Training complete.")
