
#Aqui será realizado o treino em Random Forest (um por alvo) usando
#os dados já limpos de games.db.

#Precisa do venv ativo para rodar (.\.venv\Scripts\Activate):
#python ml/train_models.py

#Isso gera os seguintes arquivos em ml/:
#    ml/model_sales_global.pkl
#    ml/model_sales_na.pkl
#    ml/model_sales_pal.pkl
#    ml/model_sales_jp.pkl
#    ml/model_sales_other.pkl
#    ml/model_metascore.pkl
#e imprime MAE (mean absolute error) de validação de cada um.


# imports  
import joblib                  
import pandas as pd              
from pathlib import Path         
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


#Configurações gerais 
DB_PATH   = Path("data/processed/games.db")           
ENGINE    = create_engine(f"sqlite:///{DB_PATH}")

# Alvos
TARGETS = {
    "global_sales":  "model_sales_global.pkl",
    "na_sales":      "model_sales_na.pkl",
    "pal_sales":     "model_sales_pal.pkl",
    "jp_sales":      "model_sales_jp.pkl",
    "other_sales":   "model_sales_other.pkl",
    "metascore":     "model_metascore.pkl"
}

# *inputs* possíveis 
NUMERIC  = ["year"]                       # fica em passthrough
CATEG    = ["rating", "platform", "genres",
            "developer", "publisher"]     # serão One-Hot

#SELECT tirando linhas sem year (ou outro NA importante).
QUERY = """
SELECT
    rating, platform, genres, developer, publisher, year,
    global_sales, na_sales, pal_sales, jp_sales, other_sales,
    metascore
FROM games_dim
WHERE year IS NOT NULL
"""
df = pd.read_sql(QUERY, ENGINE)

print(f"Dados lidos: {len(df):,} linhas")


# Pré-processador: One-Hot + passthrough 
#   • OneHotEncoder transforma strings em vetores binários.
#   • handle_unknown='ignore' deixa passar categorias novas em produção.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
        ("num", "passthrough", NUMERIC)
    ]
)


#Função utilitária p/ treinar & salvar um alvo 
def train_and_save(target_col: str, outfile: Path) -> None:
    """
    Treina Random Forest para 'target_col', avalia MAE e salva em 'outfile'.
    """

    df_target = df.dropna(subset=[target_col])

    # ● Separamos features (X) e rótulo (y)
    X = df_target[CATEG + NUMERIC]
    y = df_target[target_col]

    # ● Split estratificado não faz sentido em regressão;
    #   usamos shuffle aleatório controlado por random_state.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ● Pipeline = preprocessamento (One-Hot) ➜ modelo
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf",
             RandomForestRegressor(
                 n_estimators=700,      # mais árvores → mais robusto
                 max_depth=None,        # deixa RF escolher
                 n_jobs=-1,             # usa todos os cores
                 random_state=42))
        ]
    )

    # ● Treina
    model.fit(X_train, y_train)

    # ● Avalia
    pred = model.predict(X_val)
    mae  = mean_absolute_error(y_val, pred)
    print(f"🔸 {target_col:<12} → MAE: {mae:,.3f}")

    # ● Salva .pkl
    joblib.dump(model, Path("ml") / outfile)


# 5) Loop sobre todos os alvos solicitados
for target, filename in TARGETS.items():
    # Algumas linhas podem ter NaN no alvo específico; removemos
    if df[target].isna().all():
        print(f"⚠️  {target} só tem NaNs — pulando.")
        continue
    train_and_save(target, filename)


print("🏁  Treino completo; modelos salvos em ml/*.pkl")
