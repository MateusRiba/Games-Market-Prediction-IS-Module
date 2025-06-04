from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.features.make_features_metascore import get_train_test

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "metascore.pkl"

def train():
    (X_train, X_test, y_train, y_test), transformer = get_train_test()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    # pipeline = preprocessing + modelo
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(
        steps=[("prep", transformer), ("model", model)]
    )

    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"R² na amostra de teste: {r2:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✔ Modelo salvo em {MODEL_PATH}")
    return r2

if __name__ == "__main__":
    train()
