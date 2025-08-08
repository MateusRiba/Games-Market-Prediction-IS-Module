# ------ main.py ------
import os
import pandas as pd
from data_loading import read_data
from dataset2_transformations import main_db2
from data_pipeline import preprocess_for_modeling

# Importe a função do seu módulo de treino
from train_models import train_and_evaluate_model

# Paths e leitura dos dados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATASET1_PATH = os.path.join(BASE_DIR, 'data/raw', 'games (1).csv')
DATASET2_PATH = os.path.join(BASE_DIR, 'data/raw', 'games_sales.csv')

dataset1 = read_data(DATASET1_PATH)
dataset2 = read_data(DATASET2_PATH)

if dataset1 is not None and dataset2 is not None:
    dataset1 = main_db2(dataset1, dataset2)

    if dataset1 is not None:
        OUTPUT_FILE_NAME = 'original_dataset.csv'
        OUTPUT_FILE_PATH = os.path.join(os.path.join(BASE_DIR, 'transformers'), OUTPUT_FILE_NAME)
        dataset1.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"[INFO] Dataset mesclado salvo em '{OUTPUT_FILE_PATH}'.")
else:
    print(f"[ERRO] Não foi possível carregar os datasets. Pipeline cancelada.")

# Pré-processar os dados para o modelo
dataset1 = preprocess_for_modeling(dataset1)

# Lista das variáveis alvo para treinar modelos
target_columns = ['metascore', 'NA_Sales', 'PAL_Sales', 'JP_Sales', 'Other_Sales']

# Treinar e avaliar modelo para cada target
for target in target_columns:
    model, X_test = train_and_evaluate_model(dataset1, target)
    # Opcional: você pode armazenar os modelos em uma lista ou dicionário para uso futuro
