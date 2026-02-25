# 🔬 Cancer Risk Classification

A Machine Learning project that trains and compares three classification models to predict whether a breast tumour is **malignant or benign**. This is **Project 3 of 10** in my ML learning roadmap toward computational biology research.

---

## 📌 Project Overview

| Feature | Details |
|---|---|
| Dataset | Wisconsin Breast Cancer Dataset (built into scikit-learn) |
| Samples | 569 tumours |
| Features | 30 cell nucleus measurements |
| Target | Malignant (1) vs Benign (0) |
| Techniques | Logistic Regression, Random Forest, XGBoost, AUC-ROC, 5-Fold CV |
| Libraries | `scikit-learn`, `xgboost`, `pandas`, `matplotlib`, `seaborn` |

---

## 🧠 Models Trained

### 1. Logistic Regression
A linear classifier that models the probability of malignancy using a sigmoid function. Interpretable via coefficients — positive coefficients increase malignancy probability, negative ones decrease it.

### 2. Random Forest
An ensemble of 100 decision trees, each trained on a random data subset. Final prediction = majority vote across all trees. Robust to overfitting and provides feature importance scores.

### 3. XGBoost
Sequential boosting — each new tree corrects the errors of previous trees. State-of-the-art performance on tabular data. Also provides feature importance scores.

---

## 📊 Visualisations Generated

| Plot | What It Shows |
|---|---|
| Confusion Matrices | TP, TN, FP, FN for all 3 models side by side |
| ROC Curves | AUC comparison across all 3 models |
| Model Comparison | Accuracy, Precision, Recall, F1, AUC bar chart |
| Cross-Validation | Box plot of 5-fold AUC distributions |
| RF Feature Importance | Top 15 most important features for Random Forest |
| XGBoost Feature Importance | Top 15 most important features for XGBoost |
| LR Coefficients | Directional effect of each feature on malignancy |

---

## 📂 Project Structure

```
cancer-classification/
├── cancer_classification.py          # Main script
├── plot1_confusion_matrices.png
├── plot2_roc_curves.png
├── plot3_model_comparison.png
├── plot4_cross_validation.png
├── plot5_rf_feature_importance.png
├── plot6_xgb_feature_importance.png
├── plot7_lr_coefficients.png
└── README.md
```

---

## ⚙️ Setup Instructions

**1. Clone the repository**
```bash
git clone https://github.com/Shaflovescoffee19/cancer-classification.git
cd cancer-classification
```

**2. Install dependencies**
```bash
pip3 install scikit-learn xgboost pandas matplotlib seaborn
```

**3. Run the script**
```bash
python3 cancer_classification.py
```

---

## 💡 Example Output

```
Model                     Accuracy   Recall       F1  AUC-ROC
Logistic Regression         0.9561   0.9767   0.9524   0.9920
Random Forest               0.9649   0.9535   0.9535   0.9943
XGBoost                     0.9561   0.9535   0.9524   0.9930

Best model by AUC-ROC: Random Forest (0.9943)
```

---

## 🔬 Connection to Research Proposal

This project directly mirrors **Aim 3** of a computational biology research proposal on colorectal cancer risk prediction in the Emirati population. That proposal uses:
- The same three algorithm families (elastic net = regularised logistic regression, random forest, XGBoost)
- The same evaluation metrics (AUC-ROC, balanced accuracy, Brier scores)
- 5-fold cross-validation for robust evaluation
- Feature importance as a precursor to SHAP-based interpretability

---

## 📚 What I Learned

- How **Logistic Regression** uses the sigmoid function to model probabilities
- How **Random Forest** uses bagging and feature randomness to build robust ensembles
- How **XGBoost** uses sequential boosting to correct errors iteratively
- How to interpret **confusion matrices** — the difference between FP and FN in a clinical setting
- Why **AUC-ROC** is preferred over accuracy for medical classification
- How **5-fold cross-validation** gives a more reliable performance estimate than a single split
- How **feature importance** reveals which measurements drive predictions

---

## 🗺️ Part of My ML Learning Roadmap

| # | Project | Status |
|---|---|---|
| 1 | Heart Disease EDA | ✅ Complete |
| 2 | Diabetes Data Cleaning | ✅ Complete |
| 3 | Cancer Risk Classification | ✅ Complete |
| 4 | Survival Analysis | 🔜 Next |
| 5 | Customer Segmentation | ⏳ Upcoming |
| 6 | Gene Expression Clustering | ⏳ Upcoming |
| 7 | Explainable AI with SHAP | ⏳ Upcoming |
| 8 | Counterfactual Explanations | ⏳ Upcoming |
| 9 | Multi-Modal Data Fusion | ⏳ Upcoming |
| 10 | Transfer Learning | ⏳ Upcoming |

---

## 🙋 Author

**Shaflovescoffee19** — building ML skills from scratch toward computational biology research.
