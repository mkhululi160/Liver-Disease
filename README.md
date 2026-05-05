# Liver Disease Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Baseline%20Model-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning project that predicts whether a patient has liver disease from routine blood test results — without invasive procedures.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Limitations](#limitations)
- [Next Steps](#next-steps)
- [Project Structure](#project-structure)

---

## Overview

**Problem:** Given a patient's blood results and basic demographics, can we predict whether they have liver disease?

**Why it matters:** Liver disease affects millions globally. Early detection through non-invasive screening can prioritise high-risk patients for confirmatory testing, reducing unnecessary procedures and cutting diagnostic costs.

**Approach:** Binary classification using a Random Forest model trained on 583 labelled patient records.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| File | `liver_patient_dataset.csv` |
| Patients | 583 |
| Features | 10 (blood results + demographics) |
| Target | `Liver Disease` / `No Liver Disease` |
| Class split | 71.4% Liver Disease · 28.6% No Disease |

### Feature Descriptions

| Feature | Full Name | Type |
|---------|-----------|------|
| `Age` | Patient age | Demographic |
| `Gender` | Patient gender | Demographic |
| `TB` | Total Bilirubin | Blood test |
| `DB` | Direct Bilirubin | Blood test |
| `Alkphos` | Alkaline Phosphotase | Blood test |
| `Sgpt` | Alamine Aminotransferase | Blood test |
| `Sgot` | Aspartate Aminotransferase | Blood test |
| `TP` | Total Proteins | Blood test |
| `ALB` | Albumin | Blood test |
| `A/G Ratio` | Albumin & Globulin Ratio | Blood test |

> **Note:** Age and Gender were excluded from model features. Only the 7 numerical blood markers were used for prediction.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/liver-disease-prediction.git
cd liver-disease-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib
```

---

## Usage

Open and run the notebook top to bottom:

```bash
jupyter notebook liver_nalysis_final.ipynb
```

Or run the cells in order:
1. Load and inspect data
2. Clean and preprocess
3. Train the model
4. Evaluate and visualise results

The cleaned dataset is saved automatically as `liver_clean.csv` after preprocessing.

---

## Methodology

### 1. Data Cleaning
- The `A/G Ratio` column contained `?` string entries — converted to numeric using `pd.to_numeric(errors='coerce')`
- Checked all columns for null values; no rows were dropped (all 583 records retained)
- Binary target created: `Liver Disease` → `1`, `No Liver Disease` → `0`

### 2. Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Training: 466 rows | Test: 117 rows
```

### 3. Model

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

No hyperparameter tuning was applied — this is an intentional baseline to establish a starting point.

---

## Results

### Performance Summary

| Metric | Value |
|--------|-------|
| Accuracy | 69.2% |
| Liver Disease — Precision | 0.77 |
| Liver Disease — Recall | 0.83 |
| Liver Disease — F1 | **0.80** |
| No Disease — Precision | 0.38 |
| No Disease — Recall | 0.30 |
| No Disease — F1 | **0.33** |

### Confusion Matrix

```
                  Predicted
                  No Disease    Liver Disease
Actual No Disease      9              21
Actual Liver Disease  15              72
```

### How to Read These Results

Raw accuracy (69.2%) is misleading here. Because 71% of the dataset is Liver Disease, a model that predicted *Liver Disease for every patient* would score ~74% accuracy — higher than this model. The F1 scores tell the real story:

- **Liver Disease (F1: 0.80)** — The model identifies sick patients reasonably well
- **No Disease (F1: 0.33)** — The model frequently misclassifies healthy patients as sick (21 false positives out of 30 true negatives)

---

## Feature Importance

Ranked by mean decrease in impurity across 100 trees:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Alkphos | 0.184 |
| 2 | Sgot | 0.170 |
| 3 | Sgpt | 0.167 |
| 4 | ALB | 0.131 |
| 5 | TB | 0.125 |
| 6 | TP | 0.119 |
| 7 | DB | 0.103 |

**Key finding:** Liver enzyme markers (Alkphos, Sgot, Sgpt) are the strongest predictors — outperforming bilirubin levels (TB, DB), which were expected to rank higher. This suggests that bile duct and liver cell inflammation markers are more discriminative for this patient population.

> ⚠️ Feature importance shows correlation, not causation. These rankings are specific to this dataset and model, and have not been validated externally.

---

## Limitations

| Limitation | Detail |
|------------|--------|
| Class imbalance | 71% of records are Liver Disease — inflates accuracy and biases the model toward positive predictions |
| No Disease performance | F1 of 0.33 means the model is unreliable for ruling out disease |
| No external validation | Results have only been evaluated on a held-out split of the same dataset |
| Baseline only | No tuning, no ensembling, no cross-validation — results represent a starting point |
| Outliers unaddressed | Extreme bilirubin values in the raw data were not treated |

---

## Next Steps

- [ ] Address class imbalance using SMOTE or `class_weight='balanced'`
- [ ] Add cross-validation (k-fold) for more robust evaluation
- [ ] Test logistic regression as a simpler, interpretable baseline
- [ ] Tune hyperparameters (`max_depth`, `min_samples_split`, `n_estimators`)
- [ ] Handle outliers in bilirubin columns
- [ ] Validate on data from a separate hospital or population
- [ ] Convert output to a clinical risk score rather than a hard binary prediction

---

## Project Structure

```
liver-disease-prediction/
│
├── liver_nalysis_final.ipynb    # Main notebook — full pipeline
├── liver_patient_dataset.csv    # Raw dataset
├── liver_clean.csv              # Cleaned dataset (generated)
└── README.md
```

---

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, numpy
- **Modelling:** scikit-learn
- **Visualisation:** matplotlib

---

*This project is a learning exercise and baseline model. It is not intended for clinical use.*
