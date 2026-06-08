def pick_best_config(results: dict, weights=None):
    """Simple agent-like scorer to select an oversampling configuration.

    results: {"method": {"recall":..., "f1":..., "gmean":...}, ...}
    weights: optional dict like {"recall":0.4, "f1":0.3, "gmean":0.3}
    Returns (best_method_name, ranked_list[(name,score),...])
    """
    if weights is None:
        weights = {"recall": 1/3, "f1": 1/3, "gmean": 1/3}

    scored = []
    for name, res in results.items():
        score = (
            weights["recall"] * res["recall"] +
            weights["f1"] * res["f1"] +
            weights["gmean"] * res["gmean"]
        )
        scored.append((name, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored
