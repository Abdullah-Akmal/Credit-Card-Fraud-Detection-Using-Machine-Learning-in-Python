# Credit Card Fraud Detection Using Machine Learning

## Problem Statement
Credit card fraud presents a significant risk to financial institutions, leading to substantial monetary losses each year. The inherent imbalance in transaction data—where fraudulent events are exceedingly rare—impedes the ability of standard machine learning algorithms to detect anomalies with high accuracy. This project addresses these challenges by constructing a robust fraud-detection pipeline: it performs thorough EDA to understand data distributions, applies feature scaling and SMOTE oversampling to mitigate class imbalance, and evaluates multiple classifiers (Logistic Regression, Random Forest, XGBoost) using metrics such as precision, recall, F1-score, and ROC-AUC. The ultimate goal is to develop a reproducible, high-performance model capable of reliably identifying fraudulent transactions in real-world settings.

---

## Tools & Libraries Used
- Python  
- pandas, NumPy – Data loading and manipulation  
- seaborn, matplotlib – Visualization  
- scikit-learn – Machine learning models and evaluation
  - LogisticRegression, RandomForestClassifier  
  - classification_report, roc_auc_score, confusion_matrix  
  - train_test_split, StandardScaler  
- imbalanced-learn – SMOTE for handling class imbalance  
- XGBoost – Gradient boosting classifier for performance on imbalanced data

---

## Dataset
- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Contains 284,807 transactions with only 492 labeled as fraudulent  
- All features except 'Time' and 'Amount' are anonymized

---

## Methodology / Steps

1. Load and Inspect the Dataset  
   - Reviewed shape, null values, class distribution, and feature types

2. Exploratory Data Analysis  
   - Visualized class imbalance and distribution of key features

3. Data Preprocessing  
   - Used StandardScaler on 'Time' and 'Amount'  
   - Applied SMOTE to oversample the minority class (fraudulent transactions)

4. Model Training  
   - Logistic Regression  
   - Random Forest Classifier  
   - XGBoost Classifier

5. Model Evaluation  
   - Confusion Matrix  
   - Precision, Recall, and F1-score  
   - ROC-AUC Curve  
   - Classification Report

---

## Outcome
- SMOTE significantly improved model sensitivity to fraudulent cases  
- Random Forest and XGBoost outperformed Logistic Regression on recall and ROC-AUC  
- Developed a scalable and interpretable fraud detection pipeline  
- The methodology can be extended to other fraud or anomaly detection use cases

---

## How to Run
1. Clone the repository  
2. Install required libraries (scikit-learn, pandas, matplotlib, seaborn, imbalanced-learn, xgboost)  
3. Open the notebook: `Credit Card Fraud Detection Model.ipynb`  
4. Run all cells in Jupyter Notebook or Google Colab

---

