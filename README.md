# Thesis Fraud Detection Project (Reproducible Experiments)

This project runs reproducible experiments for imbalanced credit card fraud detection:
- Oversampling: RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
- Classifier: RandomForest
- Evaluation: Stratified 5-fold CV
- Metrics: Recall, F1-score, G-mean
- Optional: 5x2cv paired t-test (Dietterich)

## 1) Setup
Install Python 3.10+ (3.9+ usually works) and run:

```bash
pip install -r requirements.txt
```

## 2) Data
Download `creditcard.csv` from Kaggle (ULB credit card fraud dataset) and place it here:

`data/creditcard.csv`

## 3) Run experiments
```bash
python -m src.main
```

Outputs:
- `outputs/oversampler_comparison.csv`
- `outputs/adasyn_levels.csv`
- Printed ranking from the simple "agent" scorer (optional)

## 4) Notes
- This code is designed to match a common research design in the literature: oversampling **only on the training fold** (via an imblearn Pipeline).
- Results can vary slightly depending on random seed and library versions.
