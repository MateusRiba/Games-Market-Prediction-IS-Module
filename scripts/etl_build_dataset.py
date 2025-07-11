#ETL para gerar data/processed/games.db e parquet de backup
#Tratamento de dados.

import pandas as pd
from slugify import slugify #Slugs são nomes únicos e amigáveis para URLs ou chaves
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

# Padronização de nomes (isso foi uma correção)
# SQLite ignora maiúsc/minúsc.  Para evitar colisões,
# - colocamos tudo em minúsculas,
# - e para colunas homônimas vindas de fontes diferentes
#   damos sufixos claros (_mc = Metacritic, _vg = VGChartz).
def normalize_cols(cols):
    new_cols = []
    seen = {}
    for c in cols:
        base = c.lower()
        if base in seen:
            # Já vimos esse nome → gera sufixo incremental (ou use sua própria regra)
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")   # ex.: publisher_2
        else:
            seen[base] = 1
            new_cols.append(base)                     # primeira ocorrência
    return new_cols

merged_df.columns = normalize_cols(merged_df.columns)

#Chekagem das colunas
print("Colunas após merge e normalização:")
print(merged_df.columns.tolist())

#Para manter apenas um slug
merged_df = merged_df.rename(columns={"slug_g": "slug"})
merged_df = merged_df.drop(columns=["slug_s"])

# Lista de colunas consideradas ruído para o modelo
DROP_COLS = [
    "description",
    "metascore_count",
    "userscore_count",
    "platform_metascores",
    "basename",
    "vgchartz_score",
    "critic_score",
    "user_score",
    "last_update",
    "url",
    "vgchartzscore",
    "img_url"
]

merged_df = merged_df.drop(columns=DROP_COLS)


#Realizando uma limpeza de tipos

numeric_cols = [
    "metascore", "metascore_count", "userscore", "userscore_count",
    "VGChartz_Score", "Critic_Score", "User_Score",
    "Total_Shipped", "Global_Sales", "NA_Sales",
    "PAL_Sales", "JP_Sales", "Other_Sales"
]
for col in numeric_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df["releasedate"] = pd.to_datetime(
    merged_df["releasedate"], errors="coerce")
merged_df["year"] = merged_df["releasedate"].dt.year.fillna(
    merged_df["year"])   # Usa ano do VGChartz se releaseDate faltar

# Remoção de duplicatas
merged_df = merged_df.drop_duplicates(subset="slug")

#gerando colunas finais:

merged_df.insert(0, "game_id", range(1, len(merged_df) + 1))

#gravando em SQLlite

engine = create_engine(f"sqlite:///{OUT_DB}")
with engine.begin() as conn:
    merged_df.to_sql("games_dim", conn, if_exists="replace", index=False)
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS idx_slug  ON games_dim(slug);")
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS idx_genre ON games_dim(genres);")
    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS idx_year  ON games_dim(year);")
    
#backup Parquet + amostra para testes
#merged_df.to_parquet("data/processed/games_dim.parquet", index=False)

# garante que o diretório tests/data exista
SAMPLE_DIR = Path("tests/data")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)  # cria recursivamente, se precisar

# salva 300 linhas aleatórias para testes de CI
merged_df.sample(300, random_state=42).to_csv(
    SAMPLE_DIR / "sample_games.csv",
    index=False
)


merged_df.sample(300, random_state=42)\
         .to_csv("tests/data/sample_games.csv", index=False)

print(f"ETL concluído {len(merged_df):,} linhas gravadas em {OUT_DB}")