"""
train_model.py
--------------
Loads data/loan_data.csv, preprocesses it, trains a Random Forest
classifier for Loan Approval Prediction, prints evaluation metrics,
and saves the pipeline to model/loan_model.pkl.

Run:
    python train_model.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")
REPORT_IMG_DIR = os.path.join(BASE_DIR, "report_images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_IMG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────────
print("[1/6] Loading dataset …")
if not os.path.exists(DATA_PATH):
    sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 3. Basic cleaning
# ─────────────────────────────────────────────
print("[2/6] Cleaning data …")

# Drop Loan_ID if present (identifier, not a feature)
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)

# Standardise target column name
target_candidates = ["Loan_Status", "loan_status", "LoanStatus"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    sys.exit("ERROR: could not find target column (expected 'Loan_Status').")
df.rename(columns={target_col: "Loan_Status"}, inplace=True)

# Encode target: Y→1, N→0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
df.dropna(subset=["Loan_Status"], inplace=True)

print(f"      Class distribution:\n{df['Loan_Status'].value_counts()}")

# ─────────────────────────────────────────────
# 4. Feature engineering
# ─────────────────────────────────────────────
print("[3/6] Encoding features …")

# Identify column types
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = df.select_dtypes(include=["number"]).columns.tolist()
if "Loan_Status" in numerical_cols:
    numerical_cols.remove("Loan_Status")

print(f"      Categorical: {categorical_cols}")
print(f"      Numerical  : {numerical_cols}")

# Label-encode categorical columns in-place before the sklearn pipeline
# (keeps things simple and transparent)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("Unknown")
    df[col] = le.fit_transform(df[col])

# Save encoded column order for the Flask API
feature_cols = [c for c in df.columns if c != "Loan_Status"]

# ─────────────────────────────────────────────
# 5. Build sklearn Pipeline
# ─────────────────────────────────────────────
X = df[feature_cols]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Numerical sub-pipeline: impute median → scale
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, feature_cols),
])

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

# ─────────────────────────────────────────────
# 6. Cross-validation
# ─────────────────────────────────────────────
print("[4/6] Cross-validation …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"      CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. Final training + evaluation
# ─────────────────────────────────────────────
print("[5/6] Training final model …")
clf_pipeline.fit(X_train, y_train)

y_pred   = clf_pipeline.predict(X_test)
y_proba  = clf_pipeline.predict_proba(X_test)[:, 1]

acc      = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)

print("\n══════════════════════════════════════════")
print("           EVALUATION METRICS              ")
print("══════════════════════════════════════════")
print(f"  Test Accuracy : {acc:.4f}")
print(f"  ROC-AUC Score : {roc_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set", fontsize=14)
plt.tight_layout()
cm_path = os.path.join(REPORT_IMG_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n  Confusion matrix saved → {cm_path}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#3b82d4", lw=2, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=14)
ax.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(REPORT_IMG_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  ROC curve saved        → {roc_path}")

# Feature importances
importances = clf_pipeline.named_steps["classifier"].feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.head(10).plot(kind="bar", color="#3b82d4", ax=ax)
ax.set_title("Top-10 Feature Importances", fontsize=14)
ax.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fi_path = os.path.join(REPORT_IMG_DIR, "feature_importances.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"  Feature importances    → {fi_path}")

# ─────────────────────────────────────────────
# 8. Persist model + metadata
# ─────────────────────────────────────────────
print("\n[6/6] Saving model …")
model_artifact = {
    "pipeline": clf_pipeline,
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "accuracy": acc,
    "roc_auc": roc_auc,
}
joblib.dump(model_artifact, MODEL_PATH)
print(f"      Model saved → {MODEL_PATH}")
print("\n✅  Training complete.")
