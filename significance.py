import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline

def _error_rate(y_true, y_pred) -> float:
    return float(np.mean(y_true != y_pred))

def _fit_predict(X_train, y_train, X_test, sampler, model):
    if sampler is None:
        pipe = SkPipeline([("model", clone(model))])
    else:
        pipe = ImbPipeline([("sampler", sampler), ("model", clone(model))])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)

def paired_5x2cv_ttest(X, y, sampler_a, sampler_b, model, random_state: int = 42):
    """Dietterich 5x2cv paired t-test on error rate.

    Returns t_stat (you can compare to t critical with df=5 at alpha=0.05).
    """
    rng = np.random.RandomState(random_state)
    diffs = []
    vars_ = []

    for _ in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(rng.randint(0, 10_000_000)))
        splits = list(skf.split(X, y))

        tr1, te1 = splits[0]
        yhat_a_1 = _fit_predict(X.iloc[tr1], y.iloc[tr1], X.iloc[te1], sampler_a, model)
        yhat_b_1 = _fit_predict(X.iloc[tr1], y.iloc[tr1], X.iloc[te1], sampler_b, model)
        p1 = _error_rate(y.iloc[te1].values, yhat_a_1) - _error_rate(y.iloc[te1].values, yhat_b_1)

        tr2, te2 = splits[1]
        yhat_a_2 = _fit_predict(X.iloc[tr2], y.iloc[tr2], X.iloc[te2], sampler_a, model)
        yhat_b_2 = _fit_predict(X.iloc[tr2], y.iloc[tr2], X.iloc[te2], sampler_b, model)
        p2 = _error_rate(y.iloc[te2].values, yhat_a_2) - _error_rate(y.iloc[te2].values, yhat_b_2)

        diffs.append((p1, p2))
        vars_.append((p1 - p2) ** 2)

    p11 = diffs[0][0]
    denom = np.sqrt(np.mean(vars_)) if np.mean(vars_) > 0 else 1e-12
    return float(p11 / denom)
