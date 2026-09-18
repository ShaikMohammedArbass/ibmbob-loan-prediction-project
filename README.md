# 🏦 Loan Approval Prediction

A full-stack machine learning application that predicts whether a bank loan application will be **approved or rejected** — powered by a Random Forest classifier, served via a Flask REST API, and visualised through a Streamlit web interface.

---

## 📁 Project Structure

```
loan_prediction/
├── data/
│   └── loan_data.csv                        ← Dataset (place here)
├── model/
│   └── loan_model.pkl                       ← Trained model (auto-generated)
├── backend/
│   └── app.py                               ← Flask REST API
├── frontend/
│   └── ui.py                                ← Streamlit web UI
├── report_images/                           ← EDA + model performance charts (PNG)
├── train_model.py                           ← Model training script
├── generate_charts.py                       ← EDA chart generation script
├── Loan_Approval_Prediction_Report.docx     ← Project report
├── requirements.txt                         ← Python dependencies
└── README.md                                ← This file
```

---

## 📊 Dataset

**Source:** [[Kaggle — Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset))

| Column | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0 / 1 / 2 / 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income ($) |
| CoapplicantIncome | Numerical | Co-applicant monthly income ($) |
| LoanAmount | Numerical | Loan amount (thousands $) |
| Loan_Amount_Term | Numerical | Repayment term (months) |
| Credit_History | Binary | 1 = good, 0 = bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Loan_Status | **Target** | Y = Approved, N = Rejected |

614 rows · 12 features · binary classification

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│   frontend/ui.py   ·  http://localhost:8501             │
│   • Applicant form   • EDA charts   • About tab         │
└─────────────────────┬───────────────────────────────────┘
                      │  HTTP POST /predict
┌─────────────────────▼───────────────────────────────────┐
│                    Flask Backend API                     │
│   backend/app.py   ·  http://localhost:5000             │
│   GET /health  ·  GET /model/info  ·  POST /predict     │
└─────────────────────┬───────────────────────────────────┘
                      │  joblib.load()
┌─────────────────────▼───────────────────────────────────┐
│               model/loan_model.pkl                       │
│   sklearn Pipeline: Imputer → Scaler → RandomForest     │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

> **Requires:** Python 3.10 or later

```bash
# 1. Clone or copy the project
git clone <your-repo-url>
cd loan_prediction

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Copy loan_data.csv into the data/ folder:
#    loan_prediction/data/loan_data.csv
```

---

## 🚀 Run Instructions

All commands are run from the `loan_prediction/` project root.

### Step 1 — Generate EDA charts

```bash
python generate_charts.py
```

Saves PNG files to `report_images/`:
- `credit_history_vs_approval.png`
- `applicant_income_dist.png`
- `loan_amount_dist.png`
- `approval_by_property_area.png`
- `correlation_heatmap.png`
- `loan_status_pie.png`

### Step 2 — Train the model

```bash
python train_model.py
```

Outputs evaluation metrics to the console and saves:
- `model/loan_model.pkl` — trained scikit-learn pipeline
- `report_images/confusion_matrix.png`
- `report_images/roc_curve.png`
- `report_images/feature_importances.png`

### Step 3 — Start the Flask API

Open **Terminal 1**:

```bash
python backend/app.py
# API available at http://localhost:5000
```

Verify with:
```bash
curl http://localhost:5000/health
# {"model_loaded": true, "status": "ok"}
```

### Step 4 — Start the Streamlit UI

Open **Terminal 2**:

```bash
streamlit run frontend/ui.py
# UI available at http://localhost:8501
```

---

## 🔌 API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model/info`
```json
{ "accuracy": 0.8211, "features": [...], "roc_auc": 0.8634 }
```

### `POST /predict`

**Request body:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

**Response:**
```json
{
  "label": "Approved",
  "prediction": 1,
  "probability": 0.87
}
```

---

## 🤖 Model Details

| Item | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 |
| max_depth | 10 |
| class_weight | balanced |
| Imputation | Median (numerical), 'Unknown' (categorical) |
| Scaling | StandardScaler |
| CV Strategy | StratifiedKFold (5 folds) |
| Test Accuracy | ~82% |
| ROC-AUC | ~0.86 |

---

## 🌍 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOAN_API_BASE` | `http://localhost:5000` | Flask API base URL used by the Streamlit UI |

---

## 📄 Report

A professional Word document report (`Loan_Approval_Prediction_Report.docx`) is included in the project root. It contains:

- Executive Summary with key results table
- Dataset overview with feature descriptions
- EDA findings (credit history, income, loan amount, property area)
- Data preprocessing pipeline details
- Model architecture and hyperparameters
- Performance metrics and chart references
- System architecture and deployment instructions

---

## 🛠️ Tech Stack

| Layer | Library/Tool |
|---|---|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Visualisation | matplotlib, seaborn |
| Backend API | Flask |
| Frontend UI | Streamlit |
| Report Generation | python-docx (via Bob office tools) |

---

## 📜 License

MIT — free to use, modify, and distribute.
