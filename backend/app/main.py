from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path

from .api import filters        

from .api import debug

app = FastAPI(title="Analisador de Mercado de Games API")

# Cada router
app.include_router(filters.router)
app.include_router(debug.router)  

#Importando modelos treinados

MODELS_PATH = Path("ml")

MODELS = {
    "global_sales":  joblib.load(MODELS_PATH / "model_sales_global.pkl"),
    "na_sales":      joblib.load(MODELS_PATH / "model_sales_na.pkl"),
    "pal_sales":     joblib.load(MODELS_PATH / "model_sales_pal.pkl"),
    "jp_sales":      joblib.load(MODELS_PATH / "model_sales_jp.pkl"),
    "other_sales":   joblib.load(MODELS_PATH / "model_sales_other.pkl"),
    "metascore":     joblib.load(MODELS_PATH / "model_metascore_boost.pkl"),
}

ALLOWED_TARGETS = set(MODELS.keys())

#Logica de entrada e saída de dados

class PredictRequest(BaseModel):
    rating:     str | None = Field(None, description="ESRB rating, ex: 'E10+'")
    platform:   list[str] | str | None = Field(
        None, description="Plataforma única ou lista delas"
    )
    genres:     str | None = None
    developer:  str | None = None
    publisher:  str | None = None
    year:       int | None = None
    #vendas regionais podem entrar como hints pro metascore 
    global_sales:  float | None = None
    na_sales:      float | None = None
    pal_sales:     float | None = None
    jp_sales:      float | None = None
    other_sales:   float | None = None

#Endpoint de predição

@app.post("/predict/{target}")
def predict(target: str, payload: PredictRequest):
    """
    `target` deve ser uma das chaves de ALLOWED_TARGETS.
    O corpo JSON pode omitir qualquer feature.
    """
    if target not in ALLOWED_TARGETS:
        raise HTTPException(
            status_code=404,
            detail=f"Target '{target}' não suportado. Use: {', '.join(ALLOWED_TARGETS)}"
        )

    model = MODELS[target]

    # Conversão de payload para dict para DataFrame 
    data = payload.dict()

    # Se plataforma vier como lista, transforma em string separada por '|'
    if isinstance(data["platform"], list):
        data["platform"] = "|".join(data["platform"])

    X = pd.DataFrame([data])

    # previsão
    y_hat = model.predict(X)[0]

    #resposta
    return {
        "target": target,
        "prediction": round(float(y_hat), 3)
    }

#Filters para o frontend
from sqlalchemy import create_engine, text

DB_URL = "sqlite:///data/processed/games.db"
engine = create_engine(DB_URL)

@app.get("/filters")
def list_filters():
    with engine.begin() as conn:
        genres  = conn.execute(text("SELECT DISTINCT genres FROM games_dim")).scalars().all()
        ratings = conn.execute(text("SELECT DISTINCT rating FROM games_dim")).scalars().all()
        platforms = conn.execute(text("SELECT DISTINCT platform FROM games_dim")).scalars().all()
    return {
        "genres":   sorted(g for g in genres if g),
        "ratings":  sorted(r for r in ratings if r),
        "platforms":sorted(p for p in platforms if p)
    }