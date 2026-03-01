# 📊 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning project that predicts whether a bank customer
will churn, built with Python, scikit-learn, XGBoost, and Streamlit.

---

## 🖥️ App Preview

> Prediction Page + Analytics Dashboard built with Streamlit

---

## 🎯 Problem Statement

Customer churn is one of the biggest challenges in banking. Losing a customer
costs 5–10x more than retaining one. This project builds a churn prediction
system that identifies at-risk customers early so the business can take
proactive retention action.

---

## 🏗️ Project Structure
```
customer-churn-prediction/
│
├── data/
│   └── churn.csv                  ← Dataset (from Kaggle)
├── notebooks/
│   └── churn_model_training.ipynb ← EDA + Training pipeline
├── models/                        ← Saved artifacts (auto-generated)
├── preprocessing.py               ← Encoding + scaling logic
├── train.py                       ← Model training script
├── predict.py                     ← Inference logic
├── app.py                         ← Streamlit app
├── requirements.txt
└── README.md
```

---

## 🔬 Models Trained & Compared

| Model               | Accuracy | Recall | F1     | ROC-AUC |
|---------------------|----------|--------|--------|---------|
| XGBoost             | 85.4%    | 64.4%  | 64.2%  | **0.867** |
| Random Forest       | 84.5%    | 58.2%  | 60.5%  | 0.852   |
| Decision Tree       | 75.9%    | 73.2%  | 55.3%  | 0.827   |
| Logistic Regression | 72.2%    | 68.3%  | 49.9%  | 0.772   |

✅ **Best Model: XGBoost** — selected based on ROC-AUC score.

---

## 🧠 Key Findings

- **Active membership** is the strongest churn predictor
- **Number of products** — customers with only 1 product churn most
- **Age** — younger customers are more likely to churn
- **Germany** has significantly higher churn rate than France/Spain
- **Low balance** customers are at higher risk

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Nirvighna04/Customer-Churn-Prediction.git
cd customer-churn-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `churn.csv` inside the `data/` folder.
Dataset available on [Kaggle](https://www.kaggle.com/).

### 5. Train the model
```bash
python train.py
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| pandas, numpy | Data manipulation |
| matplotlib, seaborn | Visualization |
| scikit-learn | ML models + preprocessing |
| XGBoost | Best performing model |
| imbalanced-learn | SMOTE oversampling |
| joblib | Model serialization |
| Streamlit | Web app |

---

## 📄 License

This project is licensed under the MIT License.

---
