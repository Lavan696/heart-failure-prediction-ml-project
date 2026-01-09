# Heart Failure Prediction – Explainable Machine Learning Pipeline

An end-to-end **machine learning classification project** that predicts the risk of heart disease using clinical features, with a strong emphasis on **data preprocessing, class imbalance handling, threshold optimization, and model explainability**.

This project goes beyond accuracy-focused modeling by incorporating **clinical risk trade-offs**, **probability calibration**, and **SHAP-based interpretability**, reflecting real-world healthcare ML practices.

---

## Problem Statement

Heart disease remains one of the leading causes of mortality worldwide.  
Early and accurate risk prediction can significantly improve clinical decision-making and patient outcomes.

The objective of this project is to:
- Predict whether a patient is at risk of heart disease
- Handle class imbalance responsibly
- Optimize decision thresholds based on precision–recall trade-offs
- Provide transparent model explanations suitable for clinical interpretation

---

## Dataset

- **Source:** Kaggle – *Heart Failure Prediction* (by fedesoriano)
- **Target variable:** `HeartDisease` (0 = No, 1 = Yes)
- **Features:** Demographic, clinical, and test-based attributes

>  The dataset (`heart.csv`) is **not included** in this repository to comply with Kaggle’s data usage policy.

### How to obtain the dataset
1. Visit the Kaggle dataset page:  
   **Heart Failure Prediction – fedesoriano**
2. Download `heart.csv`
3. Place the file in the **root directory** of this project

---

## Project Highlights

- Custom **scikit-learn compatible FeatureProcessor**
- Robust handling of **imbalanced data** using SMOTE
- Hyperparameter tuning with **ROC-AUC optimization**
- **Threshold tuning** using cross-validated precision–recall analysis
- Comprehensive model evaluation (classification + calibration)
- **Explainable AI (SHAP)** for global and local interpretability
- Deployment-ready **model bundle serialization**

---

## Modeling Pipeline

### 1. Exploratory Data Analysis
- Target distribution analysis
- Gender-wise risk comparison
- Numerical feature distributions
- Categorical feature frequency analysis

### 2. Feature Engineering
A custom `FeatureProcessor` handles:
- Binary encoding (e.g., Sex, Exercise Angina)
- Ordinal encoding (ST_Slope)
- One-hot encoding (Chest Pain, ECG results)
- Binning of continuous variables (Age, Cholesterol)
- Deterministic feature ordering for model stability

### 3. Class Imbalance Handling
- SMOTE applied **only on training data**
- SMOTE embedded inside cross-validation pipelines to avoid leakage

### 4. Model Training
- **Random Forest Classifier**
- Hyperparameter tuning using `GridSearchCV`
- Optimization metric: **ROC-AUC**

### 5. Threshold Optimization
Instead of using a default 0.5 threshold:
- Cross-validated probability predictions are generated
- Precision–Recall trade-offs analyzed
- Decision threshold selected to balance:
  - High recall (minimizing false negatives)
  - Controlled precision (reducing false positives)

This reflects real-world healthcare risk modeling priorities.

---

## Model Performance (Test Set)

The final optimized Random Forest model was evaluated on a held-out test set using an adjusted decision threshold derived from cross-validated precision–recall analysis.

### Test Set Metrics

| Metric | Value |
|------|------|
| Accuracy | **88.04%** |
| Precision | **90.47%** |
| Recall (Sensitivity) | **88.78%** |
| F1-Score | **89.62%** |
| ROC-AUC | **0.9417** |
| Log Loss | **0.331** |
| Cohen’s Kappa | **0.755** |
| Matthews Correlation Coefficient (MCC) | **0.755** |

### Cross-Validation Performance
- **Mean CV Accuracy:** **86.37%**
- **Best CV ROC-AUC:** **0.9434**

### Interpretation
- High **recall** ensures most high-risk patients are correctly identified
- Strong **precision** reduces false positives and unnecessary interventions
- High **ROC-AUC** indicates strong class separation
- **MCC and Cohen’s Kappa** confirm balanced performance under class imbalance

These results demonstrate a clinically meaningful balance between sensitivity
and specificity rather than raw accuracy optimization.


---

Additional diagnostic tools:
- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- Calibration Curve
- Cumulative Gain Curve
- Classification Report Heatmap

---

## Model Explainability (SHAP)

To ensure transparency and trust:
- Global feature importance using SHAP summary plots
- Feature impact distribution (beeswarm plots)
- Local explanation for individual predictions

SHAP analysis helps understand **why** the model predicts a patient as high or low risk, which is critical in clinical ML applications.

---

## Model Persistence

All components required for inference are saved together:

- FeatureProcessor
- Optimized Random Forest model
- Optimized decision threshold

Saved as a single artifact:

This enables:
- Reproducible inference
- Seamless API or batch deployment
- Consistent preprocessing at prediction time

---

## Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- SHAP
- Joblib

---

## Future Work

- FastAPI-based inference service
- Dockerized deployment
- Monitoring prediction drift
- Threshold adjustment based on clinical cost functions

---

##  Author  

**Lavan Kumar Konda**  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
