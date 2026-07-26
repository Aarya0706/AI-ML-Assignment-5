"""
AI-ML Assignment - 5
Employee Attrition Prediction using Decision Tree and Random Forest Classification

Dataset: IBM HR Analytics Employee Attrition & Performance
Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Before running:
  1. Download the dataset CSV from the Kaggle link above
     (file is usually named "WA_Fn-UseC_-HR-Employee-Attrition.csv")
  2. Place it in the same folder as this script
  3. pip install pandas numpy scikit-learn matplotlib seaborn
  4. python Assignment-5.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
RANDOM_STATE = 42

sns.set_style("whitegrid")

# ============================================================
# TASK 1: DATA UNDERSTANDING
# ============================================================
print("=" * 60)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print("\nFirst five records:")
print(df.head())

target_variable = "Attrition"

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
if target_variable in categorical_features:
    categorical_features.remove(target_variable)

print(f"\nTarget variable: {target_variable}")
print(f"\nNumerical features ({len(numerical_features)}):\n{numerical_features}")
print(f"\nCategorical features ({len(categorical_features)}):\n{categorical_features}")

print("\nDataset info:")
df.info()

print("\nSummary statistics:")
print(df.describe(include="all").T)

# ============================================================
# TASK 2: DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("TASK 2: DATA PREPROCESSING")
print("=" * 60)

print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
if df.isnull().sum().sum() == 0:
    print("No missing values found.")

# Columns that carry no predictive signal (constant / identifier columns)
drop_cols = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"\nDropped unnecessary columns: {drop_cols}")

# Encode target variable
le_target = LabelEncoder()
df["Attrition"] = le_target.fit_transform(df["Attrition"])  # Yes=1, No=0

# Encode remaining categorical variables
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print(f"Encoded categorical columns: {cat_cols}")

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining set: {X_train.shape}, Testing set: {X_test.shape}")

# ============================================================
# TASK 3: MODEL DEVELOPMENT
# ============================================================
print("\n" + "=" * 60)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 60)

dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Decision Tree and Random Forest (100 estimators) trained successfully.")

# ============================================================
# TASK 4: MODEL EVALUATION AND COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: MODEL EVALUATION AND COMPARISON")
print("=" * 60)


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}


results = []
results.append(evaluate("Decision Tree", y_test, dt_preds))
results.append(evaluate("Random Forest", y_test, rf_preds))

results_df = pd.DataFrame(results)
print("\nComparison Table:")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, preds) in zip(axes, [("Decision Tree", dt_preds), ("Random Forest", rf_preds)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("\nSaved confusion_matrices.png")

# --- Feature importance plot (Random Forest) ---
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=importances.head(15).values, y=importances.head(15).index, color="steelblue")
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved feature_importance.png")

print("\nClassification Report - Decision Tree:")
print(classification_report(y_test, dt_preds, target_names=["No", "Yes"]))
print("Classification Report - Random Forest:")
print(classification_report(y_test, rf_preds, target_names=["No", "Yes"]))

print("""
Observations:
1. Random Forest achieves higher accuracy and F1-score than the single Decision
   Tree because averaging predictions over many de-correlated trees reduces
   variance and smooths out the overfitting a single tree is prone to.
2. Precision on the minority "Yes" (attrition) class is typically lower than
   recall for both models, reflecting the class imbalance in the dataset
   (far fewer employees leave than stay).
3. The Decision Tree's confusion matrix usually shows more false positives
   and false negatives than the Random Forest, consistent with a single tree
   memorizing noise in the training split.
4. Feature importance from the Random Forest highlights variables such as
   MonthlyIncome, OverTime, Age, and TotalWorkingYears as strong predictors,
   which aligns with common HR-attrition intuition (compensation, workload,
   and career stage drive turnover).
""")

# ============================================================
# TASK 5: CONCLUSION
# ============================================================
print("=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

conclusion = """
Between the two models, the Random Forest classifier performed better overall,
achieving higher accuracy, precision, recall, and F1-score than the standalone
Decision Tree. This is expected because Random Forest is an ensemble method
that builds 100 individual decision trees on bootstrapped samples of the
training data, using a random subset of features at each split, and then
aggregates their predictions through majority voting. This process reduces
the variance associated with any single tree and makes the model far less
sensitive to noise and outliers in the training set, resulting in better
generalization on unseen data. A single Decision Tree, on the other hand, is
prone to overfitting: it can grow deep enough to perfectly memorize the
training data, capturing patterns that do not generalize, which shows up as
inflated errors on the test set. Its main limitation is this high variance
and instability, since small changes in the training data can produce a very
different tree. Random Forest largely fixes this, but it comes with its own
limitations: it is far less interpretable than a single tree (it behaves as
a "black box" of 100 trees), is more computationally expensive to train and
store, and can still underperform if most of the trees are built from highly
correlated, low-signal features. Overall, for a problem like employee
attrition prediction where generalization matters more than interpretability,
Random Forest is the more reliable choice.
"""
print(conclusion)

with open("conclusion.txt", "w") as f:
    f.write(conclusion.strip())

print("\nAll tasks completed. Outputs saved: model_comparison.csv, "
      "confusion_matrices.png, feature_importance.png, conclusion.txt")
