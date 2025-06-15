# Credit Card Fraud Detection

## Description  
This notebook builds a machine learning model to detect fraudulent credit card transactions using an imbalanced dataset. It explores data distribution, applies preprocessing techniques, and compares various classification algorithms to handle class imbalance and improve detection accuracy.  
**[Dataset Source – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

## Objectives  
- Understand the structure and imbalance of credit card transaction data  
- Apply preprocessing techniques for feature scaling and resampling  
- Build and evaluate machine learning models to detect fraud  
- Compare model performance using precision, recall, F1-score, and ROC-AUC  

## Methodology / Steps  
- Load and inspect the dataset (shape, class balance, sample entries)  
- Perform exploratory data analysis (EDA) and visualize feature distributions  
- Use techniques like SMOTE to balance class distribution  
- Apply feature scaling using StandardScaler  
- Train classification models including Logistic Regression, Random Forest, and XGBoost  
- Evaluate models using confusion matrix, ROC-AUC curve, and classification reports  

## Tools & Libraries Used  
Python, pandas, NumPy, seaborn, matplotlib, scikit-learn (LogisticRegression, RandomForestClassifier, classification_report, roc_auc_score, train_test_split, StandardScaler), imblearn (SMOTE), XGBoost
