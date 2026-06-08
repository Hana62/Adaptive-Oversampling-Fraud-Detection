from pathlib import Path
import pandas as pd

from .config import Config
from .experiments import load_data, compare_oversamplers, adasyn_levels
from .agent import pick_best_config

def _to_table(results: dict):
    rows = []
    for name, res in results.items():
        rows.append({
            "method": name,
            "recall": res["recall"],
            "f1": res["f1"],
            "gmean": res["gmean"],
        })
    return pd.DataFrame(rows).sort_values(by="gmean", ascending=False)

def main():
    cfg = Config()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    X, y = load_data(cfg.data_path, cfg.target_col)

    # 1) Compare oversamplers
    results = compare_oversamplers(
        X, y,
        n_splits=cfg.n_splits,
        random_state=cfg.random_state,
        n_estimators=cfg.n_estimators,
    )
    df = _to_table(results)
    df.to_csv(out_dir / "oversampler_comparison.csv", index=False)
    print("\n=== Oversampler comparison (saved to outputs/oversampler_comparison.csv) ===")
    print(df.to_string(index=False))

    # 2) Agent selection (optional)
    best, ranking = pick_best_config({k: v for k, v in results.items()})
    print("\n=== Simple agent selection ===")
    print("Best method:", best)
    print("Ranking (method, score):")
    for name, score in ranking:
        print(f"  {name}: {score:.6f}")

    # 3) ADASYN levels
    levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    lvl_results = adasyn_levels(
        X, y, levels,
        n_splits=cfg.n_splits,
        random_state=cfg.random_state,
        n_estimators=cfg.n_estimators,
    )
    df2 = pd.DataFrame([{
        "target_fraud_percent": k,
        "recall": v["recall"],
        "f1": v["f1"],
        "gmean": v["gmean"],
    } for k, v in lvl_results.items()]).sort_values(by="target_fraud_percent")
    df2.to_csv(out_dir / "adasyn_levels.csv", index=False)
    print("\n=== ADASYN levels (saved to outputs/adasyn_levels.csv) ===")
    print(df2.to_string(index=False))

if __name__ == "__main__":
    main()
