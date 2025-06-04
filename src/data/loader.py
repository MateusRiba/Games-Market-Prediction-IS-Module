# src/data/loader.py
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_csv(filename: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Carrega um arquivo CSV da pasta data/raw.

    Parameters
    ----------
    filename : str
        Nome do arquivo dentro de data/raw (ex.: 'games_metadata.csv')
    read_csv_kwargs : Any
        Qualquer argumento extra passado para pandas.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Verifique o nome ou caminho.")
    return pd.read_csv(path, **read_csv_kwargs)
