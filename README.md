# 🔬 Cancer Risk Classification -> Comparing Three ML Algorithms

Classification is the most common task in clinical machine learning, given a set of measurements, predict which category a patient belongs to. This project builds and compares three classification algorithms of increasing complexity on a well-studied medical dataset, exploring the trade-off between model simplicity and predictive power and learning when complexity actually helps.

---

## 📌 Project Snapshot

| | |
|---|---|
| **Dataset** | Wisconsin Breast Cancer Dataset (sklearn built-in) |
| **Records** | 569 tumour samples |
| **Features** | 30 measurements (radius, texture, perimeter, area, smoothness, etc.) |
| **Target** | Malignant vs Benign (binary) |
| **Models** | Logistic Regression · Random Forest · XGBoost |
| **Libraries** | `scikit-learn` · `xgboost` · `pandas` · `matplotlib` · `seaborn` |

---

## 🗂️ The Dataset

The Wisconsin Breast Cancer dataset contains measurements computed from digitised images of fine needle aspirate (FNA) biopsies. Each of 30 features describes a different geometric property of the cell nuclei present in the image, mean, worst, and standard error versions of 10 base measurements. The target is whether the tumour was ultimately diagnosed as malignant or benign.

This dataset is notable for being genuinely linearly separable, the two classes can be distinguished with high accuracy by a simple linear boundary, which makes it an ideal benchmark for comparing algorithm complexity.

---

## 🤖 Models Trained

### Logistic Regression
The simplest classifier, learns a linear decision boundary by fitting a weighted sum of features through a sigmoid function. Fast, interpretable, and a strong baseline whenever the data is approximately linearly separable. Coefficients directly indicate each feature's contribution to the prediction.

### Random Forest
An ensemble of 100 decision trees, each trained on a random sample of data and features. Individual trees overfit, the ensemble corrects this by averaging predictions across trees. Captures non-linear relationships and provides feature importance scores without requiring scaling.

### XGBoost
Sequential boosted trees, each new tree corrects the residual errors of all previous trees. State-of-the-art on tabular data, handles missing values, includes built-in regularisation, and provides its own feature importance metric. More hyperparameters to tune but often the highest-performing model on structured data.

---

## 📊 Evaluation Framework

- **5-fold stratified cross-validation** on training set for robust performance estimates
- **Held-out test set** for final unbiased evaluation
- **Metrics reported:** Accuracy, Precision, Recall, F1, AUC-ROC, and Confusion Matrix

| Model | Accuracy | Recall | F1 | AUC |
|-------|----------|--------|-----|-----|
| Logistic Regression | 96.5% | 92.9% | 0.951 | 0.996 |
| Random Forest | 97.4% | 92.9% | 0.963 | 0.993 |
| XGBoost | 96.5% | 90.5% | 0.950 | 0.993 |

---

## 📈 Visualisations Generated

| File | Description |
|------|-------------|
| `plot1_confusion_matrices.png` | Confusion matrix for all three models |
| `plot2_roc_curves.png` | ROC curves with AUC annotations |
| `plot3_model_comparison.png` | Side-by-side metric comparison bar chart |
| `plot4_cv_distributions.png` | Cross-validation score distributions per model |
| `plot5_rf_feature_importance.png` | Random Forest feature importances |
| `plot6_xgb_feature_importance.png` | XGBoost feature importances |
| `plot7_lr_coefficients.png` | Logistic Regression feature coefficients |

---

## 🔍 Key Findings

**Logistic Regression wins on AUC (0.996) despite being the simplest model.** This is the most important lesson from this project, complexity does not equal performance. The breast cancer dataset is nearly linearly separable, and logistic regression is perfectly suited to this structure. Adding trees and boosting rounds does not help because the fundamental relationship is already linear.

**Top 5 features by Random Forest importance:**
1. `worst area` (0.1514) — largest cell area in the sample
2. `worst concave points` (0.1265) — worst concavity measure
3. `worst radius` (0.0935)
4. `worst perimeter` (0.0836)
5. `mean concave points` (0.0811)

All five are "worst" measurements, the largest values observed in the sample rather than averages. The most extreme cell in the biopsy is more diagnostically informative than the average cell, which is clinically meaningful for malignancy detection.

---

## 📂 Repository Structure

```
cancer-classification/
├── cancer_classification.py
├── plot1_confusion_matrices.png
├── plot2_roc_curves.png
├── plot3_model_comparison.png
├── plot4_cv_distributions.png
├── plot5_rf_feature_importance.png
├── plot6_xgb_feature_importance.png
├── plot7_lr_coefficients.png
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/Shaflovescoffee19/cancer-classification.git
cd cancer-classification
pip3 install scikit-learn xgboost pandas matplotlib seaborn
python3 cancer_classification.py
```

---

## 📚 Skills Developed

- Understanding logistic regression, Random Forest, and XGBoost -> architectures, assumptions, and trade-offs
- 5-fold stratified cross-validation and why it produces more reliable estimates than a single train/test split
- Reading and interpreting confusion matrices -> TP, TN, FP, FN and their clinical implications
- AUC-ROC -> what it measures, why it is preferred over accuracy for medical classification, and how to compare models using it
- Feature importance from tree-based models vs coefficients from linear models
- The most common modelling mistake: assuming more complex = better performance

---

## 🗺️ Learning Roadmap
_**Project 3 of 10** _ -> a structured series building from data exploration through to advanced ML techniques.

| # | Project | Focus |
|---|---------|-------|
| 1 | Heart Disease EDA | Exploratory analysis, visualisation |
| 2 | Diabetes Data Cleaning | Missing data, outliers, feature engineering |
| 3 | **Cancer Risk Classification** ← | Supervised learning, model comparison |
| 4 | Survival Analysis | Time-to-event modelling, Cox regression |
| 5 | Customer Segmentation | Clustering, unsupervised learning |
| 6 | Gene Expression Clustering | High-dimensional data, heatmaps |
| 7 | Explainable AI with SHAP | Model interpretability |
| 8 | Counterfactual Explanations | Actionable predictions |
| 9 | Multi-Modal Data Fusion | Stacking, ensemble methods |
| 10 | Transfer Learning | Neural networks, domain adaptation |
