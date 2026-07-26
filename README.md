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
| Accuracy  | 0.7823 | 0.8435 |
| Precision | 0.3191 | 0.5455 |
| Recall    | 0.3191 | 0.1277 |
| F1-Score  | 0.3191 | 0.2069 |

Random Forest wins clearly on accuracy and precision — it makes far fewer
false-positive "will leave" predictions. However, it has notably lower
recall than the Decision Tree here, meaning it misses more of the employees
who actually do leave. This is a common effect of class imbalance (attrition
"Yes" is the minority class): Random Forest's averaging pushes it toward the
majority class ("No"), trading recall for precision. Depending on the
business goal, this trade-off matters — if the priority is catching as many
at-risk employees as possible (high recall), the Decision Tree's balance of
precision/recall might actually be more useful despite its lower accuracy;
if the priority is trusting flagged predictions (high precision), Random
Forest is the better choice.

## Conclusion
On this run, the Random Forest classifier achieved higher accuracy (0.84 vs.
0.78) and precision (0.55 vs. 0.32) than the Decision Tree, but the Decision
Tree had noticeably higher recall (0.32 vs. 0.13) on the minority "attrition
= Yes" class. Overall, Random Forest is the stronger general-purpose model:
by aggregating 100 trees trained on bootstrapped samples with random feature
subsets, it reduces the variance and overfitting that a single Decision Tree
is prone to, which is why its accuracy and precision are higher. Its lower
recall here is a side effect of the dataset's class imbalance (far fewer
employees leave than stay) — ensembling tends to favor the majority class
unless the imbalance is explicitly corrected (e.g., with class weighting or
resampling). The Decision Tree's main limitation is instability: it can grow
deep enough to memorize the training data, so small changes in the training
set can produce a very different tree and inconsistent generalization. Random
Forest's main limitation is reduced interpretability and higher computational
cost, since it behaves as a "black box" ensemble of 100 trees rather than a
single readable decision path. Which model is "better" ultimately depends on
whether the business cares more about trusting flagged predictions
(favoring Random Forest's precision) or catching as many at-risk employees
as possible (favoring the Decision Tree's recall).

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