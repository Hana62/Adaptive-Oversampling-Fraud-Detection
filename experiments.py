import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

from .metrics import compute_metrics

def load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:10]} ...")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def make_model(random_state: int, n_estimators: int):
    # max_features="sqrt" corresponds to sqrt(p)
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

def run_cv_experiment(X, y, sampler, model, n_splits: int, random_state: int) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if sampler is None:
            pipe = SkPipeline([("model", model)])
        else:
            # IMPORTANT: oversampling is applied only on training folds via pipeline
            pipe = ImbPipeline([("sampler", sampler), ("model", model)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        fold_metrics.append(compute_metrics(y_test, y_pred))

    out = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    out["folds"] = fold_metrics
    return out

def compare_oversamplers(X, y, n_splits: int, random_state: int, n_estimators: int):
    model = make_model(random_state=random_state, n_estimators=n_estimators)

    experiments = {
        "No oversampling": None,
        "Random oversampling": RandomOverSampler(random_state=random_state),
        "SMOTE": SMOTE(k_neighbors=5, random_state=random_state),
        "ADASYN": ADASYN(n_neighbors=5, random_state=random_state),
        "Borderline-SMOTE": BorderlineSMOTE(k_neighbors=5, random_state=random_state),
    }

    results = {}
    for name, sampler in experiments.items():
        results[name] = run_cv_experiment(
            X=X, y=y,
            sampler=sampler,
            model=model,
            n_splits=n_splits,
            random_state=random_state,
        )
    return results

def adasyn_levels(X, y, levels, n_splits: int, random_state: int, n_estimators: int):
    """Evaluate ADASYN with different target fraud proportions.

    In imblearn, sampling_strategy is minority/majority after resampling.
    If target fraud proportion is p:
        minority/(minority+majority) = p  => minority/majority = p/(1-p)
    """
    model = make_model(random_state=random_state, n_estimators=n_estimators)

    out = {}
    for p in levels:
        ratio = p / (1.0 - p)
        sampler = ADASYN(n_neighbors=5, sampling_strategy=ratio, random_state=random_state)
        out[f"{int(p*100)}%"] = run_cv_experiment(
            X=X, y=y,
            sampler=sampler,
            model=model,
            n_splits=n_splits,
            random_state=random_state,
        )
    return out
