from pathlib import Path
import pandas as pd

def load_qpm(root):
    table = pd.read_csv(Path(root) / "model" / "quarterly_training.csv")
    table.index = pd.PeriodIndex(table.pop("period"), freq="Q")
    assert table.index.is_unique and table.index.is_monotonic_increasing
    assert not table.isna().any().any()
    return table
