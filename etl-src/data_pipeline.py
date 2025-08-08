# --- Arquivo: data_pipeline.py ---
import pandas as pd
import numpy as np
from datetime import datetime
from utils import map_sentiment, map_rating, impute_sentiment, process_platforms_and_metascores, process_publisher_column, process_genre_column

from config import sentiment_dict, rating_dict, broad_platform_mapping, developer_name_mapping, publisher_name_mapping, important_publishers, important_developers

from sklearn.impute import SimpleImputer # Para imputação de NaNs


def preprocess_for_modeling(df):
    """
    Executa a pipeline de pré-processamento para o dataset que já foi mesclado.
    """
    print(f"[INÍCIO] Pré-processamento do Dataset...")
    df = df.copy()

    # 1. Apagando colunas de description e ID
    df.drop(columns=['description', 'id'], inplace=True, errors='ignore')
    print(f"[INFO] Colunas 'description' e 'id' removidas.")

    # 2. Mapeando colunas de sentimento e rating
    df = map_sentiment(df, ['metascore_sentiment', 'userscore_sentiment'], sentiment_dict)
    df = map_rating(df, 'rating', rating_dict)
    print(f"[INFO] Colunas 'sentiment' e 'rating' mapeadas para numéricas.")

    # 3. Imputando valores nulos em userscore_sentiment
    df = impute_sentiment(df, 'metascore', 'userscore_sentiment')
    print(f"[INFO] Valores nulos em 'userscore_sentiment' imputados.")

    # 4. Transformando a data
    df['releaseDate'] = pd.to_datetime(df['releaseDate'])
    df_date_features = pd.DataFrame(index=df.index)
    df_date_features['release_year'] = df['releaseDate'].dt.year
    df_date_features['release_month'] = df['releaseDate'].dt.month
    df_date_features['release_day'] = df['releaseDate'].dt.day
    df_date_features['release_day_of_week'] = df['releaseDate'].dt.dayofweek
    df_date_features['release_quarter'] = df['releaseDate'].dt.quarter
    
    df = pd.concat([df, df_date_features.drop(columns=['release_date_original'], errors='ignore')], axis=1)
    print(f"[INFO] Features de data extraídas e adicionadas.")

    # 5. Processando Platforms, Developer, Publisher, Genre
    #    (A lógica completa para estas colunas entra aqui, como nas nossas conversas anteriores)
    #    Por exemplo:
    #    print("\n[PASSO 5] Iniciando tratamento de 'developer'...")
    #    df_ohe_dev, dev_cleaned_col = process_multi_label_column(df.copy(), 'developer', 'dev', developer_name_mapping, important_developers, 10)
    #    df = pd.concat([df, df_ohe_dev], axis=1)
    #    df.drop(columns=['developer', dev_cleaned_col], inplace=True, errors='ignore')
    #    print("Tratamento de 'developer' concluído. Total de colunas após OHE:", df.shape[1])
    #    ... e assim por diante para todas as outras colunas ...

    # 6. Conversão de tipos e imputação final
    #    (A lógica final de astype e SimpleImputer entra aqui)
    
    df_broad_metascores = process_platforms_and_metascores(df.copy(), broad_platform_mapping)
    
    # ... (código que você já tem para outras colunas) ...

    # Por fim, remova as colunas originais de plataforma
    df.drop(columns=['platforms', 'platform_metascores'], inplace=True, errors='ignore')

        # 5. Processando Platforms, Developer, Publisher, Genre
    print("\n[PASSO 5] Iniciando tratamento de 'developer'...")

    # Importa aqui dentro se preferir lazy load, ou direto no topo do arquivo
    from utils import process_developer_column

    df_dev_ohe = process_developer_column(
        df.copy(),
        developer_mapping=developer_name_mapping,
        important_developers=important_developers,
        min_frequency=10,  # mínimo de aparições para não virar "outros"
        verbose=True
    )

    df = pd.concat([df, df_dev_ohe], axis=1)
    df.drop(columns=['developer'], inplace=True, errors='ignore')

    print(f"[INFO] Tratamento de 'developer' concluído. Total de colunas após OHE: {df.shape[1]}")
    
    print("[INFO] Iniciando tratamento de 'publisher'...")

    df_pub_ohe = process_publisher_column(
        df,
        publisher_col='publisher',
        publisher_name_mapping=publisher_name_mapping,
        important_publishers=important_publishers,
        min_frequency=20,
        verbose=True
    )
    df = pd.concat([df, df_pub_ohe], axis=1)
    df.drop(columns=['publisher'], inplace=True, errors='ignore')

    print(f"[INFO] Tratamento de 'publisher' concluído. Total de colunas após OHE: {df.shape[1]}")

    print("\[INFO] Iniciando tratamento de 'genres'...")

    df_genre_ohe = process_genre_column(
        df,
        genre_col='genres',
        min_frequency=10,  # ajuste o threshold conforme seu dataset
        verbose=True
    )
    df = pd.concat([df, df_genre_ohe], axis=1)
    df.drop(columns=['genres'], inplace=True, errors='ignore')

    print(f"[INFO] Tratamento de 'genres' concluído. Total de colunas após OHE: {df.shape[1]}")

    df.drop(columns=['releaseDate', 'title'], inplace=True, errors='ignore')

    print(f"[FIM] Pré-processamento do Dataset concluído. ---")

    return df
