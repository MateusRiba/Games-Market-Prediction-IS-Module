#ETL para gerar data/processed/games.db e parquet de backup
#Tratamento de dados.

import pandas as pd
from slugify import slugify
from rapidfuzz import process, fuzz
from sqlalchemy import create_engine
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DB  = Path("data/processed/games.db")

#carregar CSV
games  = pd.read_csv(RAW_DIR / "games (1).csv")     # Metacritic
sales  = pd.read_csv(RAW_DIR / "games_sales.csv")   # VGChartz

#Criando slug normalizado
games["slug"] = games["title"].apply(slugify)       # ex.: "Super-Mario-Galaxy-2"
sales["slug"] = sales["basename"].fillna(sales["Name"].apply(slugify))   # basename já vem pronto, mas garanta fallback

def fuzzy_left_join(left: pd.DataFrame, right: pd.DataFrame,
                    key_left: str="slug", key_right: str="slug",
                    threshold: int = 90) -> pd.DataFrame:
    """
    Tenta casar cada slug do `left` com o slug mais parecido do `right`.
    Se a similaridade (0-100) ≥ threshold, devolve o match; senão mantém NaN.
    
    """
    matches = []
    right_slugs = right[key_right].tolist()

    for s in left[key_left]:
        match, score, idx = process.extractOne(
            s, right_slugs, scorer=fuzz.token_sort_ratio)
        matches.append(match if score >= threshold else pd.NA)

    left["slug_match"] = matches
    merged = left.merge(
        right, left_on="slug_match", right_on=key_right, how="left",
        suffixes=("_g", "_s"))
    return merged.drop(columns=["slug_match"])

merged_df = fuzzy_left_join(games, sales)

#realizando uma limpeza de tipos

numeric_cols = [
    "metascore", "metascore_count", "userscore", "userscore_count",
    "VGChartz_Score", "Critic_Score", "User_Score",
    "Total_Shipped", "Global_Sales", "NA_Sales",
    "PAL_Sales", "JP_Sales", "Other_Sales"
]
for col in numeric_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df["releaseDate"] = pd.to_datetime(
    merged_df["releaseDate"], errors="coerce")
merged_df["Year"] = merged_df["releaseDate"].dt.year.fillna(
    merged_df["Year"])   # Usa ano do VGChartz se releaseDate faltar
merged_df = merged_df.drop_duplicates(subset="slug")

#gerando colunas finais:

merged_df.insert(0, "game_id", range(1, len(merged_df) + 1))

#gravando em SQLlite

engine = create_engine(f"sqlite:///{OUT_DB}")
with engine.begin() as conn:
    merged_df.to_sql("games_dim", conn, if_exists="replace", index=False)
    # Índices úteis para queries rápidas no FastAPI
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_slug   ON games_dim(slug);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_genre  ON games_dim(genres);")
    conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_year   ON games_dim(Year);")

#backup Parquet + amostra para testes
merged_df.to_parquet("data/processed/games_dim.parquet", index=False)
merged_df.sample(300, random_state=42)\
         .to_csv("tests/data/sample_games.csv", index=False)

print(f"ETL concluído {len(merged_df):,} linhas gravadas em {OUT_DB}")