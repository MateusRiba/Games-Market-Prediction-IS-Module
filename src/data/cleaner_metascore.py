from pathlib import Path
import pandas as pd
from src.data.loader import load_csv

OUT = Path(__file__).resolve().parents[2] / "data" / "processed" / "metascore.parquet"

def build_dataset():
    df = load_csv("games.csv")
    df = df[["title", "genres", "rating", "developer", "metascore"]]

    # Drop linhas sem nota ou sem gênero
    df = df.dropna(subset=["metascore", "genres"])

    # Converte metascore para numérico
    df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")
    df = df.dropna(subset=["metascore"])

    df.to_parquet(OUT, index=False)
    print(f"✔ Dataset salvo em {OUT} com {len(df)} linhas")

if __name__ == "__main__":
    build_dataset()
