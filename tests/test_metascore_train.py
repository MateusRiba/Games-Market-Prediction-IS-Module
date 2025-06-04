from pathlib import Path
from src.models.train_metascore import train

def test_model_training(tmp_path, monkeypatch):
    # Redireciona models/ para pasta temporária
    monkeypatch.setattr("src.models.train_metascore.MODEL_PATH",
                        tmp_path / "model.pkl")
    r2 = train()
    assert r2 > 0.10                 # ajusta o limiar conforme o dataset
    assert (tmp_path / "model.pkl").exists()
