from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd, joblib
from pathlib import Path
from sqlalchemy import create_engine, text

from .api import filters, debug

app = FastAPI(title="Analisador de Mercado de Games API")

app.include_router(filters.router)
app.include_router(debug.router)

# ---------------------- Modelos ----------------------
MODELS_PATH = Path("ml")
MODELS = {
    "global_sales": joblib.load(MODELS_PATH / "model_sales_global.pkl"),
    "na_sales":     joblib.load(MODELS_PATH / "model_sales_na.pkl"),
    "pal_sales":    joblib.load(MODELS_PATH / "model_sales_pal.pkl"),
    "jp_sales":     joblib.load(MODELS_PATH / "model_sales_jp.pkl"),
    "other_sales":  joblib.load(MODELS_PATH / "model_sales_other.pkl"),
    "metascore":    joblib.load(MODELS_PATH / "model_metascore.pkl"),
}
ALLOWED_TARGETS = set(MODELS)

# ---------------------- Schema ----------------------
class PredictRequest(BaseModel):
    rating:    str | None = Field(None, description="ESRB rating (ex.: 'E10+')")

    platform:  list[str] | str | None = Field(
        None, description="Uma plataforma ou lista delas"
    )
    genres:    list[str] | str | None = Field(
        None, description="Um ou dois gêneros separados ou lista"
    )

    developer: str | None = None
    publisher: str | None = None
    year:      int | None = None

    # campos opcionais que podem servir de hint
    global_sales: float | None = None
    na_sales:     float | None = None
    pal_sales:    float | None = None
    jp_sales:     float | None = None
    other_sales:  float | None = None

# ---------------------- Endpoint ----------------------
@app.post("/predict/{target}")
def predict(target: str, payload: PredictRequest):
    if target not in ALLOWED_TARGETS:
        raise HTTPException(404, f"Target inválido. Use: {', '.join(ALLOWED_TARGETS)}")

    data = payload.model_dump()

    # normaliza listas -> string "|"
    for col in ("platform", "genres"):
        if isinstance(data[col], list):
            data[col] = "|".join(data[col])

    X = pd.DataFrame([data])
    y_hat = MODELS[target].predict(X)[0]
    return {"target": target, "prediction": round(float(y_hat), 3)}

# ---------------------- Filtros p/ frontend ----------------------
DB_URL = "sqlite:///data/processed/games.db"
engine = create_engine(DB_URL)

@app.get("/filters")
def list_filters():
    with engine.begin() as conn:
        genres     = conn.execute(text("SELECT DISTINCT genres FROM games_dim")).scalars().all()
        ratings    = conn.execute(text("SELECT DISTINCT rating FROM games_dim")).scalars().all()
        platforms  = conn.execute(text("SELECT DISTINCT platform FROM games_dim")).scalars().all()
    return {
        "genres":    sorted([g for g in genres if g]),
        "ratings":   sorted([r for r in ratings if r]),
        "platforms": sorted([p for p in platforms if p]),
    }
