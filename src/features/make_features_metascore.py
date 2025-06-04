import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "metascore.parquet"

CATEGORICAL = ["genres", "rating", "developer"]

def get_train_test():
    df = pd.read_parquet(DATA)
    y = df["metascore"]
    X = df.drop(columns=["metascore", "title"])

    transformer = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )

    return train_test_split(X, y, test_size=0.2, random_state=42), transformer
