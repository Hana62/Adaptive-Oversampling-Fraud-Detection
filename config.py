from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str = "data/creditcard.csv"
    target_col: str = "Class"
    n_splits: int = 5
    random_state: int = 42
    n_estimators: int = 200
