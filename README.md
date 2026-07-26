# AI-ML Assignment 5 — Employee Attrition Prediction

## Name- Aarya Shirsath
## Reg. No.- 23BCE10884
## Application number- IN26011842

## Objective
Predict whether an employee is likely to leave the organization (attrition) using
demographic, professional, and work-related attributes, and compare a **Decision
Tree Classifier** against a **Random Forest Classifier (100 estimators)**.

## Dataset
IBM HR Analytics Employee Attrition & Performance
Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

(Dataset is not included in this repository — download it from the link above
and place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the project root before
running the script.)

## Libraries Used
- pandas, numpy — data loading and manipulation
- scikit-learn — preprocessing, Decision Tree / Random Forest models, metrics
- matplotlib, seaborn — visualizations (confusion matrices, feature importance)

## Methodology
1. **Data Understanding** — loaded the dataset, inspected the first five records,
   identified numerical vs. categorical features and the target variable
   (`Attrition`), and reviewed dataset info/summary statistics.
2. **Data Preprocessing** — checked for missing values, dropped non-informative
   columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`),
   label-encoded all categorical variables including the target, and split the
   data 80/20 into training and test sets (stratified on the target).
3. **Model Development** — trained a `DecisionTreeClassifier` and a
   `RandomForestClassifier(n_estimators=100)` on the same training set.
4. **Model Evaluation** — compared both models on Accuracy, Precision, Recall,
   and F1-Score; generated confusion matrices for each model and a feature
   importance plot for the Random Forest.
5. **Conclusion** — summarized which model performed better and why.

## Results
Running `Assignment-5.py` produces:
- Console output with dataset overview, preprocessing summary, and evaluation metrics
- `model_comparison.csv` — side-by-side metrics table for both models
- `confusion_matrices.png` — confusion matrices for Decision Tree and Random Forest
- `feature_importance.png` — top 15 feature importances from the Random Forest
- `conclusion.txt` — written conclusion

## Model Comparison
| Metric    | Decision Tree | Random Forest |
|-----------|---------------|----------------|
| Accuracy  | *see model_comparison.csv after running* | |
| Precision | | |
| Recall    | | |
| F1-Score  | | |

Random Forest is expected to outperform the single Decision Tree on most
metrics, since averaging over 100 de-correlated trees reduces variance and
overfitting relative to any single tree.

## Conclusion
Random Forest generally outperforms a standalone Decision Tree because it
aggregates predictions from many trees trained on bootstrapped samples with
random feature subsets, which reduces variance and improves generalization.
A single Decision Tree is prone to overfitting and is highly sensitive to
small changes in the training data. Random Forest trades away some of that
interpretability and computational efficiency, and can still be limited if
built from highly correlated features, but for a prediction task like
employee attrition — where generalizing to unseen employees matters more
than reading a single tree's logic — it is the more reliable model.

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python Assignment-5.py
```

## Bonus (Hyperparameter Tuning)
Tuning `n_estimators` on the Random Forest (e.g., 50 vs. 100 vs. 200 trees)
typically shows diminishing returns past ~100 trees — accuracy improves
initially as more trees reduce variance, then plateaus, while training time
keeps increasing roughly linearly.
