# ============================================================
# PROJECT 3: Cancer Risk Classification
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the Wisconsin Breast Cancer dataset
#   2. Explores and prepares the data
#   3. Trains 3 models: Logistic Regression, Random Forest, XGBoost
#   4. Evaluates each with AUC-ROC, accuracy, precision, recall, F1
#   5. Uses 5-fold cross-validation for robust evaluation
#   6. Plots ROC curves for all 3 models
#   7. Compares model performance in a summary chart
#   8. Plots feature importance for Random Forest and XGBoost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# ===========================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ===========================================================
# The Wisconsin Breast Cancer dataset contains measurements
# from digitized images of fine needle aspirates of breast
# masses. Each row is a tumour. 30 features describe the
# cell nucleus: size, shape, texture, smoothness, etc.

data = load_breast_cancer()

# Convert to DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
# Note: 0 = malignant (cancer), 1 = benign (no cancer)
# We will flip this so 1 = malignant for intuitive reading
df["target"] = 1 - df["target"]
# Now: 1 = Malignant, 0 = Benign

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"  Total samples  : {df.shape[0]}")
print(f"  Total features : {df.shape[1] - 1}")
print(f"  Malignant (1)  : {(df['target'] == 1).sum()} ({(df['target']==1).mean()*100:.1f}%)")
print(f"  Benign    (0)  : {(df['target'] == 0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
print()
print("Feature categories:")
print("  mean_*     : Mean value of each characteristic")
print("  error_*    : Standard error (spread/consistency)")
print("  worst_*    : Largest (worst) value of each characteristic")
print()

# ===========================================================
# STEP 2: PREPARE FEATURES AND TARGET
# ===========================================================

X = df.drop("target", axis=1)
y = df["target"]

# Train/test split — 80/20, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 60)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Train malignant  : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Test malignant   : {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print()

# Scale features — important for Logistic Regression
# (Random Forest and XGBoost don't need it, but it doesn't hurt)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit ONLY on train
X_test_scaled = scaler.transform(X_test)          # Apply same scale to test

# ===========================================================
# STEP 3: TRAIN ALL 3 MODELS
# ===========================================================

print("=" * 60)
print("STEP 3: TRAINING MODELS")
print("=" * 60)

# ── Model 1: Logistic Regression ──────────────────────────
# C = inverse of regularisation strength (higher C = less regularisation)
# max_iter = max number of iterations for the solver to converge
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("  Logistic Regression  : trained")

# ── Model 2: Random Forest ────────────────────────────────
# n_estimators = number of trees (more = better but slower)
# max_depth = how deep each tree can grow (None = unlimited)
# min_samples_split = minimum samples required to split a node
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=2, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)   # No scaling needed for RF
print("  Random Forest        : trained")

# ── Model 3: XGBoost ──────────────────────────────────────
# n_estimators = number of boosting rounds
# learning_rate = how much each tree contributes (lower = more conservative)
# max_depth = depth of each tree
# subsample = fraction of data used for each tree (prevents overfitting)
xgb_model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=4,
    subsample=0.8, random_state=42,
    eval_metric="logloss", verbosity=0
)
xgb_model.fit(X_train, y_train)  # XGBoost handles raw features fine
print("  XGBoost              : trained")
print()

# ===========================================================
# STEP 4: EVALUATE ALL MODELS
# ===========================================================

print("=" * 60)
print("STEP 4: MODEL EVALUATION ON TEST SET")
print("=" * 60)

models = {
    "Logistic Regression": (lr_model, X_test_scaled),
    "Random Forest":       (rf_model, X_test),
    "XGBoost":             (xgb_model, X_test),
}

results = {}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]  # Probability of malignant

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "AUC-ROC": auc,
        "y_prob": y_prob, "y_pred": y_pred
    }

    print(f"\n  {name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")

print()

# ===========================================================
# STEP 5: 5-FOLD CROSS-VALIDATION
# ===========================================================
# Cross-validation gives a more honest performance estimate
# by testing on every sample at least once.

print("=" * 60)
print("STEP 5: 5-FOLD CROSS-VALIDATION (AUC-ROC)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    "Logistic Regression": (lr_model, X_train_scaled),
    "Random Forest":       (rf_model, X_train.values),
    "XGBoost":             (xgb_model, X_train.values),
}

cv_results = {}
for name, (model, X_cv) in cv_models.items():
    scores = cross_val_score(model, X_cv, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name}")
    print(f"    Fold AUCs : {[f'{s:.3f}' for s in scores]}")
    print(f"    Mean AUC  : {scores.mean():.4f} ± {scores.std():.4f}")
    print()

# ===========================================================
# STEP 6: CONFUSION MATRICES
# ===========================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(models.keys())

for i, (name, (model, X_eval)) in enumerate(models.items()):
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                ax=axes[i], cbar=False)
    axes[i].set_title(f"{name}\nAUC = {results[name]['AUC-ROC']:.3f}",
                      fontweight="bold")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

    # Annotate TP, TN, FP, FN
    axes[i].text(0.5, -0.18,
        f"TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}",
        transform=axes[i].transAxes, ha="center", fontsize=9, color="gray")

fig.suptitle("Confusion Matrices — All 3 Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot1_confusion_matrices.png")
plt.close()
print("Saved: plot1_confusion_matrices.png")

# ===========================================================
# STEP 7: ROC CURVES — ALL 3 MODELS
# ===========================================================
# The ROC curve shows the trade-off between catching all
# malignant cases (high recall) vs avoiding false alarms
# (high specificity) at every possible threshold.

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0", "#DD8452", "#55A868"]

for i, (name, (model, X_eval)) in enumerate(models.items()):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
    auc = results[name]["AUC-ROC"]
    ax.plot(fpr, tpr, color=colors[i], lw=2.5,
            label=f"{name} (AUC = {auc:.3f})")

# Random baseline
ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Baseline (AUC = 0.500)")

ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
ax.set_title("ROC Curves — Logistic Regression vs Random Forest vs XGBoost",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_roc_curves.png")
plt.close()
print("Saved: plot2_roc_curves.png")

# ===========================================================
# STEP 8: MODEL COMPARISON BAR CHART
# ===========================================================

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
model_names_list = list(results.keys())

comparison_data = {
    metric: [results[m][metric] for m in model_names_list]
    for metric in metrics
}

x = np.arange(len(metrics))
width = 0.25
colors_bar = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, color) in enumerate(zip(model_names_list, colors_bar)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width,
                  label=name, color=color, edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_model_comparison.png")
plt.close()
print("Saved: plot3_model_comparison.png")

# ===========================================================
# STEP 9: CROSS-VALIDATION SCORE DISTRIBUTION
# ===========================================================

fig, ax = plt.subplots(figsize=(9, 5))
cv_data = [cv_results[name] for name in cv_results]
bp = ax.boxplot(cv_data, patch_artist=True,
                labels=list(cv_results.keys()),
                notch=False)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("AUC-ROC Score", fontsize=12)
ax.set_title("5-Fold Cross-Validation AUC Distribution",
             fontsize=13, fontweight="bold")
ax.set_ylim(0.88, 1.01)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_cross_validation.png")
plt.close()
print("Saved: plot4_cross_validation.png")

# ===========================================================
# STEP 10: FEATURE IMPORTANCE — RANDOM FOREST
# ===========================================================

rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
rf_importance.plot(kind="barh", ax=ax, color="#DD8452", edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Feature Importance Score", fontsize=12)
ax.set_title("Top 15 Feature Importances — Random Forest",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot5_rf_feature_importance.png")
plt.close()
print("Saved: plot5_rf_feature_importance.png")

# ===========================================================
# STEP 11: FEATURE IMPORTANCE — XGBOOST
# ===========================================================

xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
xgb_importance.plot(kind="barh", ax=ax, color="#55A868", edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Feature Importance Score", fontsize=12)
ax.set_title("Top 15 Feature Importances — XGBoost",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot6_xgb_feature_importance.png")
plt.close()
print("Saved: plot6_xgb_feature_importance.png")

# ===========================================================
# STEP 12: LOGISTIC REGRESSION COEFFICIENTS
# ===========================================================
# Logistic Regression is the only model where coefficients
# directly tell you the direction of each feature's effect.
# Positive coefficient = feature increases malignancy probability.
# Negative coefficient = feature decreases malignancy probability.

lr_coef = pd.Series(
    lr_model.coef_[0],
    index=data.feature_names
).sort_values(ascending=False)

top_pos = lr_coef.head(8)
top_neg = lr_coef.tail(8)
lr_top = pd.concat([top_pos, top_neg]).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
colors_coef = ["#DD8452" if c > 0 else "#4C72B0" for c in lr_top.values]
lr_top.plot(kind="barh", ax=ax, color=colors_coef, edgecolor="white")
ax.invert_yaxis()
ax.axvline(x=0, color="black", linewidth=1.2, linestyle="--")
ax.set_xlabel("Coefficient Value", fontsize=12)
ax.set_title("Logistic Regression Coefficients\n(Orange = increases malignancy risk, Blue = decreases)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plot7_lr_coefficients.png")
plt.close()
print("Saved: plot7_lr_coefficients.png")

# ===========================================================
# FINAL SUMMARY
# ===========================================================

print()
print("=" * 60)
print("PROJECT 3 COMPLETE — FINAL SUMMARY")
print("=" * 60)
print()
print(f"  {'Model':<25} {'Accuracy':>9} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
print(f"  {'-'*25} {'-'*9} {'-'*8} {'-'*8} {'-'*9}")
for name in results:
    r = results[name]
    print(f"  {name:<25} {r['Accuracy']:>9.4f} {r['Recall']:>8.4f} "
          f"{r['F1']:>8.4f} {r['AUC-ROC']:>9.4f}")

print()
best = max(results, key=lambda x: results[x]["AUC-ROC"])
print(f"  Best model by AUC-ROC : {best} ({results[best]['AUC-ROC']:.4f})")
print()
print("  Most important features (Random Forest):")
for i, (feat, score) in enumerate(rf_importance.head(5).items(), 1):
    print(f"    {i}. {feat} ({score:.4f})")
print()
print("  7 plots saved.")
print("  Ready to push to GitHub!")
print("=" * 60)
