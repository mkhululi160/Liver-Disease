🫀 Liver Disease Prediction
Predicting liver disease from routine blood test results using a Random Forest classifier.
📌 Problem Statement
Given blood test results and basic patient demographics, can we predict whether a patient has liver disease — without invasive procedures?
Business value: Non-invasive screening to prioritise high-risk patients for confirmatory testing, reducing unnecessary procedures and costs.
📊 Dataset
Attribute
Detail
Source
liver_patient_dataset.csv
Rows
583 patients
Features
10 (blood results + demographics)
Target
Liver Disease / No Liver Disease
Class split
71.4% Liver Disease, 28.6% No Disease
Features used in model: TB, DB, Alkphos, Sgpt, Sgot, TP, ALB
🔧 Workflow
Code
Data Cleaning — Converted A/G Ratio from string to numeric (handled ? entries), checked for nulls across all columns
Target Encoding — Liver Disease → 1, No Liver Disease → 0
Train/Test Split — 80/20 split, random_state=42
Model — RandomForestClassifier(n_estimators=100, random_state=42)
📈 Results
Metric
Score
Accuracy
69.2%
Liver Disease F1
0.80
No Disease F1
0.33
Confusion Matrix:
Code
The model detects Liver Disease well (recall 0.83) but struggles with No Disease cases — expected given the 71% class imbalance.
🔍 Feature Importance
Rank
Feature
Importance
1
Alkphos
0.184
2
Sgot
0.170
3
Sgpt
0.167
4
ALB
0.131
5
TB
0.125
6
TP
0.119
7
DB
0.103
Key finding: Liver enzyme levels (Alkphos, Sgot, Sgpt) are the strongest predictors — more so than bilirubin levels (TB/DB).
🛠 Tech Stack
Python 3
pandas, numpy
scikit-learn (RandomForestClassifier)
matplotlib
🚀 Next Steps
[ ] Address class imbalance (SMOTE or class weights)
[ ] Test logistic regression as interpretable baseline
[ ] Handle bilirubin outliers in raw data
[ ] Validate on data from a different hospital
[ ] Hyperparameter tuning (max_depth, min_samples_split)
📁 Files
Code
