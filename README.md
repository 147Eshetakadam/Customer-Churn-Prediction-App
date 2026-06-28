# 📡 ChurnIQ — Customer Churn Prediction System

A production-ready ML web application for predicting telecom customer churn.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

### 4. Login
| Username | Password |
|----------|----------|
| admin    | churn2025 |
| analyst  | telecom123 |

---

## 📁 Project Structure

```
churn_app/
├── app.py                  # Flask application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── utils/
│   ├── preprocessor.py    # Data cleaning, encoding, normalization
│   ├── trainer.py         # ML model training and evaluation
│   └── visualizer.py      # Chart data generation
│
├── model/                 # Saved models (auto-created)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Layout with sidebar
│   ├── login.html
│   ├── index.html         # Overview/home
│   ├── upload.html        # Dataset upload
│   ├── train.html         # Model training
│   ├── predict.html       # Single prediction
│   └── dashboard.html     # Analytics
│
├── static/
│   ├── css/
│   ├── js/
│   └── img/
│
└── data/
    └── sample_telco.csv   # Sample dataset for testing
```

---

## 🧭 Usage Guide

### Step 1 — Upload Data
- Navigate to **Upload Data**
- Upload your Telco CSV (or use `data/sample_telco.csv`)
- View dataset statistics and preview

### Step 2 — Train Model
- Navigate to **Train Model**
- Choose algorithm: **Random Forest** or **Logistic Regression**
- Set train/test split
- Click **Train Model**
- View Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Step 3 — Predict
- Navigate to **Predict**
- Fill in customer details (or click **Load Sample**)
- Click **Predict Churn**
- See: Yes/No prediction + probability score + risk level

### Step 4 — Dashboard
- Navigate to **Dashboard**
- View: Churn distribution, Feature importance, Confusion matrix, Tenure analysis

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.x + Flask |
| ML | scikit-learn (Logistic Regression, Random Forest) |
| Data | Pandas, NumPy |
| Charts | Chart.js |
| UI | HTML5, CSS3, Vanilla JS |
| Storage | CSV + pickle (local) |

---

## 📊 Dataset Format

Compatible with [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn).

Required columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`,
`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`,
`PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

---

## 🔬 ML Pipeline

1. Load CSV → Drop `customerID`
2. Convert `TotalCharges` to numeric (handles spaces)
3. Fill missing values with median
4. Encode binary Yes/No columns to 0/1
5. One-hot encode multi-class categoricals
6. Standardize numerical features (StandardScaler)
7. Train with `class_weight='balanced'` for imbalanced data
8. Evaluate on held-out test set

---

## 📈 Expected Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | ~82–87% | ~70–78% |
| Logistic Regression | ~78–82% | ~65–72% |

*(Results depend on dataset size and distribution)*

---

## 📝 Project Info

- **Title**: Customer Churn Prediction System
- **Institution**: Sinhgad Institute of Technology, Lonavala
- **Guide**: Prof. N.K. Patil
- **Department**: Computer Engineering
- **Year**: 2025–26
