# Heart Failure Prediction using Machine Learning

## Project Overview
This project focuses on predicting **heart disease risk** using structured clinical data and machine learning techniques.  
The goal is to build a **robust, interpretable, and clinically reliable classification model**, with special attention to **class imbalance**, **feature engineering**, and **model explainability**.

---

## Dataset
- **Dataset**: Heart Failure Prediction Dataset  
- **Source**: Kaggle  
- **File**: `heart.csv`

The dataset contains demographic, clinical, and diagnostic features such as age, sex, chest pain type, cholesterol levels, ECG results, exercise-induced angina, and more.  
The target variable indicates the **presence or absence of heart disease**.(Heart Disease (0 = No, 1 = Yes))

---

This project aims to:
- Handle imbalance using **SMOTE**
- Apply **leakage-safe feature engineering**
- Optimize decision thresholds beyond default settings
- Provide **transparent explanations** using SHAP values

---

## Feature Engineering (Custom Transformer)
A **custom scikit-learn–compatible transformer (`FeatureProcessor`)** was implemented to ensure reusable, clean, and pipeline-safe preprocessing.

### Key transformations include:
- **Binary encoding** (e.g., `Sex`, `ExerciseAngina`)
- **Ordinal encoding** (`ST_Slope`)
- **One-hot encoding**:
  - `ChestPainType`
  - `RestingECG`
- **Binning of continuous variables**:
  - Age
  - Cholesterol
- Strict separation of `fit` and `transform` steps to **prevent data leakage**

This transformer ensures consistent preprocessing across training, validation, and test data.

---

## Handling Class Imbalance (SMOTE)
To address severe class imbalance:
- **SMOTE (Synthetic Minority Over-sampling Technique)** was applied
- Oversampling was performed **only on training data**
- Integrated correctly using an **`imblearn.pipeline.Pipeline`**

This ensures SMOTE is applied **inside cross-validation folds**, preventing optimistic bias and data leakage.

---

## Model Training
- **Model**: Random Forest Classifier
- **Hyperparameter tuning**: GridSearchCV
- **Primary tuning metric**: ROC-AUC
- **Cross-validation**: 8-fold CV
- **Learning curves** used to analyze bias–variance behavior

---

## Model Evaluation
The model was evaluated using **multiple complementary metrics** to reflect real-world clinical reliability:

| Metric                         | Value      |
|--------------------------------|------------|
| Cross-Validation Mean Accuracy | **86.37%** |
| ROC-AUC (Best CV)              | **0.9434** |
| Test Accuracy                  | **88.04%** |
| Precision                      | **90.47%** |
| Recall                         | **88.78%** |
| F1-Score                       | **89.62%** |
| ROC-AUC                        | **0.9417** |
| Log Loss                       |  **0.331** |
| Cohen’s Kappa Score            |  **0.755** |
| Matthews Corr Coeff (MCC)      |  **0.755** |


---

## Threshold Optimization
Instead of using the default probability threshold (0.5):
- Precision–Recall curves were analyzed
- A **custom decision threshold** was selected
- Balanced **high recall** with acceptable precision

This improves clinical usefulness by reducing false negatives.

---

## Model Interpretability (SHAP)
**SHAP (SHapley Additive Explanations)** was used to explain model behavior at both global and local levels:

- **Global feature importance** (bar plot)
- **Feature impact distribution** (beeswarm plot)
- **Local explanations** for individual patient predictions

SHAP enhances transparency and trust, which is critical for healthcare ML systems.

---

## Diagnostics & Visualizations
The project includes extensive visual analysis:
- Feature distributions
- Class imbalance analysis
- Learning curves
- Precision–Recall curve
- ROC curve
- Calibration curve
- Confusion matrix
- Classification report heatmap
- Cumulative gain curve
- Feature importance plots
- SHAP explanations

---

## Tech Stack

- **Programming Language**: Python
- **Dataset Source**: Kaggle (Heart Failure Prediction Dataset)
- **Data Manipulation & Analysis**: NumPy, Pandas  
- **Data Visualization**: Matplotlib, Seaborn  
- **Machine Learning**: scikit-learn  
- **Imbalanced Data Handling**: imbalanced-learn (SMOTE)  
- **Model Interpretability**: SHAP (SHapley Additive Explanations)   
- **Model Persistence**: joblib  

## Model Persistence
The optimized Random Forest model is saved for reuse:

- **File**: `random_forest_model.pkl`
- **Method**: `joblib`

This allows inference or deployment without retraining.

---

## Future Improvements
- Deployment as an API or dashboard

##  Author  

**Lavan Kumar Konda**  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
