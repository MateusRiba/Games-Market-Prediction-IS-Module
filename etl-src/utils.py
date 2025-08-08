# --- Arquivo: utils.py ---
import numpy as np
import pandas as pd

def map_sentiment(df, columns_to_map, mapping):
    """Mapeia colunas de sentimento de string para valores numéricos."""
    df[columns_to_map] = df[columns_to_map].apply(
        lambda col: col.map(mapping).fillna(col)
    )
    return df

def map_rating(df, column, mapping):
    """Mapeia a coluna de rating de string para valores numéricos."""
    df[column] = df[column].map(mapping).fillna(df[column])
    return df

def impute_sentiment(df, metascore_col, sentiment_col):
    """Imputa NaNs em uma coluna de sentimento usando ranges de metascore."""
    # Faixas de Valores
    conditions = [
        (df[metascore_col] <= 53),
        (df[metascore_col] > 53) & (df[metascore_col] <= 60),
        (df[metascore_col] > 60) & (df[metascore_col] <= 70),
        (df[metascore_col] > 70) & (df[metascore_col] <= 84),
        (df[metascore_col] > 84)
    ]
    choices = [1, 2, 3, 4, 5]
    mask_to_fill = df[sentiment_col].isna()
    inferred_sentiments = np.select(conditions, choices, default=np.nan)
    df.loc[mask_to_fill, sentiment_col] = inferred_sentiments[mask_to_fill]
    return df

def process_platforms_and_metascores(df, broad_platform_mapping):
    """
    Processa as colunas multi-label 'platforms' e 'platform_metascores'
    para gerar features de metascore por categoria ampla de plataforma.
    """
    df_platforms_scores = pd.DataFrame(index=df.index)
    df_platforms_scores['platforms_original'] = df['platforms']
    df_platforms_scores['platform_metascores_original'] = df['platform_metascores']

    # --- CORREÇÃO: Assegura que o tipo seja string antes de qualquer operação de string ---
    df_platforms_scores['platforms_original'] = df_platforms_scores['platforms_original'].fillna('').astype(str)
    df_platforms_scores['platform_metascores_original'] = df_platforms_scores['platform_metascores_original'].fillna('').astype(str)
    
    df_platforms_scores['platforms_list'] = df_platforms_scores['platforms_original'].str.split(',').apply(
        lambda x: [p.strip().lower().replace(' ', '_').replace('(iphone/ipad)', 'ios_iphone_ipad') for p in x if p.strip() != '']
    )
    df_platforms_scores['metascores_list'] = df_platforms_scores['platform_metascores_original'].str.split(',').apply(
        lambda x: [int(s.strip()) for s in x if s.strip().isdigit()]
    )

    df_platforms_scores['platforms_list'] = df_platforms_scores['platforms_list'].apply(
        lambda x: [broad_platform_mapping.get(p, p) for p in x]
    )

    def create_pairs(row):
        return list(zip(row['platforms_list'], row['metascores_list']))

    df_platforms_scores['platform_score_pairs'] = df_platforms_scores.apply(create_pairs, axis=1)

    df_platforms_exploded = df_platforms_scores.explode('platform_score_pairs')
    df_platforms_exploded[['platform_name_broad', 'platform_metascore_value']] = pd.DataFrame(
        df_platforms_exploded['platform_score_pairs'].tolist(), index=df_platforms_exploded.index
    )

    df_broad_metascores = df_platforms_exploded.pivot_table(
        index=df_platforms_exploded.index, columns='platform_name_broad', values='platform_metascore_value', fill_value=np.nan
    ).groupby(level=0).mean()
    df_broad_metascores.columns = ['metascore_broad_' + col for col in df_broad_metascores.columns]

    df_broad_metascores = (df_broad_metascores >= 1).astype(int)

    return df_broad_metascores

def process_developer_column(df, developer_mapping, important_developers=None, min_frequency=10, verbose=False):
    """
    Pipeline completa para tratamento da coluna 'developer':
        - Limpeza básica
        - Explosão de múltiplos devs
        - Mapeamento para nomes canônicos
        - One-hot encoding
        - Agrupamento em 'outros' se abaixo da frequência mínima ou não for importante
    
    Args:
        df (pd.DataFrame): DataFrame com a coluna 'developer'
        developer_mapping (dict): dicionário de mapeamento canônico
        important_developers (list): lista de nomes canônicos importantes a preservar
        min_frequency (int): frequência mínima para manter uma coluna OHE
        verbose (bool): exibe prints de depuração

    Returns:
        pd.DataFrame: One-hot encoding filtrado dos desenvolvedores
    """
    if 'developer' not in df.columns:
        raise ValueError("Coluna 'developer' não encontrada no DataFrame.")

    # 1. Limpeza básica
    df_dev = create_cleaned_developer_df(df)

    # 2. Explosão de multi-labels
    df_exploded = explode_developers(df_dev)

    # 3. Mapeamento para nomes canônicos
    df_exploded = map_developer_names(df_exploded, developer_mapping)

    # 4. Criação do one-hot encoding com nomes limpos
    df_ohe = create_developer_ohe(df_exploded)

    # 5. Filtragem de colunas por frequência e/ou mapeamento
    all_keys = set(developer_mapping.keys())
    canonical_names = set(developer_mapping.values())
    keys_to_preserve = canonical_names.union(important_developers or [])

    df_ohe_filtered = filter_important_developers(df_ohe, keys_to_preserve, min_frequency)

    if verbose:
        print(f"Total de desenvolvedores únicos após limpeza: {df_ohe.shape[1]}")
        print(f"Total de desenvolvedores mantidos após filtro: {df_ohe_filtered.shape[1]}")
        print("Exemplo de colunas finais:", df_ohe_filtered.columns[:5].tolist())

    return df_ohe_filtered

def summarize_developers(df):
    print("--- Número de desenvolvedores únicos ---")
    print(df['developer'].nunique())

    print("\n--- Contagem de ocorrências dos desenvolvedores (Top 20 e os últimos 5, incluindo NaNs) ---")
    print(df['developer'].value_counts(dropna=False).head(20))
    print(df['developer'].value_counts(dropna=False).tail(5))

    print("--- Primeiras 10 entradas da coluna 'developer' ---")
    print(df['developer'].head(10))

    has_comma = df['developer'].astype(str).str.contains(',').any()
    print(f"\nAlguma entrada na coluna 'developer' contém vírgula? {has_comma}")

    if has_comma:
        print("\nExemplos de entradas multi-label:")
        print(df[df['developer'].astype(str).str.contains(',')]['developer'].head(5))
        print(f"\nTotal de linhas com múltiplos desenvolvedores: {df['developer'].astype(str).str.contains(',').sum()}")
    else:
        print("\nNão foram encontradas entradas multi-label.")

def clean_developer_name_basic(name):
    name = name.lower().strip()
    name = name.replace('.', '').replace(',', '').replace("'", '').replace('!', '')
    return name if name else np.nan

def create_cleaned_developer_df(df):
    df_dev = pd.DataFrame(index=df.index)
    df_dev['developer_original'] = df['developer'].fillna('').astype(str)
    df_dev['developer_list_cleaned'] = df_dev['developer_original'].str.split(',').apply(
        lambda lst: [clean_developer_name_basic(d) for d in lst if clean_developer_name_basic(d) is not np.nan]
    )
    return df_dev

def explode_developers(df_dev):
    df_exploded = df_dev.explode('developer_list_cleaned').dropna(subset=['developer_list_cleaned'])
    return df_exploded

def map_developer_names(df_exploded, mapping):
    df_exploded['developer_canonical_name'] = df_exploded['developer_list_cleaned'].apply(
        lambda name: mapping.get(name, name)
    )
    return df_exploded

def create_developer_ohe(df_exploded):
    developer_ohe_raw = pd.get_dummies(df_exploded['developer_list_cleaned'], prefix='dev')
    developer_ohe = developer_ohe_raw.groupby(developer_ohe_raw.index).max()
    return developer_ohe

def filter_important_developers(developer_ohe, mapping_keys, threshold):
    developer_counts = developer_ohe.sum()
    cols_to_keep = []
    cols_to_other = []

    for col in developer_ohe.columns:
        name = col.replace('dev_', '')
        if name in mapping_keys or developer_counts[col] >= threshold:
            cols_to_keep.append(col)
        else:
            cols_to_other.append(col)

    if cols_to_other:
        developer_ohe['dev_other_developer'] = developer_ohe[cols_to_other].sum(axis=1)
        developer_ohe['dev_other_developer'] = (developer_ohe['dev_other_developer'] > 0).astype(int)
        cols_to_keep.append('dev_other_developer')

    return developer_ohe[cols_to_keep].astype(int)

def find_unmapped_frequent_devs(cleaned_counts, mapped_keys, ohe_columns, threshold):
    frequent = cleaned_counts[
        (cleaned_counts.index.isin(ohe_columns.str.replace('dev_', '')))
        & (~cleaned_counts.index.isin(mapped_keys))
        & (cleaned_counts >= threshold)
    ]
    return frequent.head(50)

def process_publisher_column(
    df,
    publisher_col='publisher',
    publisher_name_mapping=None,
    important_publishers=None,
    min_frequency=20,
    verbose=False
):
    """
    Processa a coluna de publisher do DataFrame para criar colunas OHE.

    Parâmetros:
    - df: DataFrame original.
    - publisher_col: nome da coluna de publisher.
    - publisher_name_mapping: dict para mapear nomes limpos para nomes canônicos.
    - important_publishers: lista de nomes canônicos a manter sempre.
    - min_frequency: frequência mínima para manter a coluna sem agrupar em "outros".
    - verbose: imprime mensagens.

    Retorna:
    - DataFrame com colunas OHE para os publishers (e coluna pub_other_publisher para raros).
    """

    if verbose:
        print(f"[INFO] Processando coluna '{publisher_col}' para publishers...")

    df_publisher_features = pd.DataFrame(index=df.index)
    df_publisher_features['publisher_original'] = df[publisher_col].fillna('').astype(str)

    def clean_publisher_name_basic(name):
        name = name.lower().strip()
        name = name.replace('.', '').replace(',', '').replace("'", '').replace('!', '')
        return name if name else np.nan

    # Split, limpeza e mapeamento
    df_publisher_features['publisher_cleaned_list'] = df_publisher_features['publisher_original'].str.split(',').apply(
        lambda list_of_pubs: [
            publisher_name_mapping.get(clean_publisher_name_basic(p), clean_publisher_name_basic(p))
            for p in list_of_pubs if clean_publisher_name_basic(p) is not np.nan
        ]
    )

    df_publisher_exploded = df_publisher_features.explode('publisher_cleaned_list')
    df_publisher_exploded = df_publisher_exploded.dropna(subset=['publisher_cleaned_list'])

    publisher_ohe_raw = pd.get_dummies(df_publisher_exploded['publisher_cleaned_list'], prefix='pub')
    publisher_ohe = publisher_ohe_raw.groupby(publisher_ohe_raw.index).max()

    publisher_counts_ohe = publisher_ohe.sum()

    current_pubs_to_keep = []
    current_pubs_to_sum_into_other = []

    for col_name_full in publisher_ohe.columns:
        pub_name_canonical = col_name_full.replace('pub_', '')
        if pub_name_canonical in important_publishers:
            current_pubs_to_keep.append(col_name_full)
        elif publisher_counts_ohe[col_name_full] >= min_frequency:
            current_pubs_to_keep.append(col_name_full)
        else:
            current_pubs_to_sum_into_other.append(col_name_full)

    publisher_ohe_working = publisher_ohe.copy()
    if current_pubs_to_sum_into_other:
        publisher_ohe_working['pub_other_publisher'] = publisher_ohe_working[current_pubs_to_sum_into_other].sum(axis=1)
        publisher_ohe_working['pub_other_publisher'] = (publisher_ohe_working['pub_other_publisher'] > 0).astype(int)

    final_cols_for_pub_selection = current_pubs_to_keep[:]
    if 'pub_other_publisher' in publisher_ohe_working.columns and 'pub_other_publisher' not in final_cols_for_pub_selection:
        final_cols_for_pub_selection.append('pub_other_publisher')

    publisher_ohe_final = publisher_ohe_working[final_cols_for_pub_selection].loc[:, ~publisher_ohe_working[final_cols_for_pub_selection].columns.duplicated()]
    publisher_ohe_final = publisher_ohe_final.astype(int)

    if verbose:
        print(f"[INFO] Processamento de publishers concluído. Colunas criadas: {publisher_ohe_final.shape[1]}")

    return publisher_ohe_final

def process_genre_column(
    df,
    genre_col='genres',
    min_frequency=5,
    verbose=False
):
    """
    Processa a coluna de gêneros do DataFrame para criar colunas OHE com filtragem de gêneros raros.

    Parâmetros:
    - df: DataFrame original.
    - genre_col: nome da coluna de gêneros.
    - min_frequency: frequência mínima para manter a coluna sem agrupar em "outros".
    - verbose: imprime mensagens.

    Retorna:
    - DataFrame com colunas OHE para gêneros (e coluna genre_other_genre para raros).
    """

    if verbose:
        print(f"[INFO] Processando coluna '{genre_col}' para gêneros...")

    df_genres_features = pd.DataFrame(index=df.index)
    df_genres_features['genres_original'] = df[genre_col].fillna('').astype(str)

    def clean_genre_name_basic(name):
        name = name.lower().strip()
        name = name.replace('.', '').replace(',', '').replace("'", '').replace('!', '')
        return name if name else np.nan

    df_genres_features['genre_list_cleaned'] = df_genres_features['genres_original'].str.split(',').apply(
        lambda list_of_genres: [
            clean_genre_name_basic(g) for g in list_of_genres if clean_genre_name_basic(g) is not np.nan
        ]
    )

    df_genres_exploded = df_genres_features.explode('genre_list_cleaned')
    df_genres_exploded = df_genres_exploded.dropna(subset=['genre_list_cleaned'])

    genre_ohe_raw = pd.get_dummies(df_genres_exploded['genre_list_cleaned'], prefix='genre')
    genre_ohe = genre_ohe_raw.groupby(genre_ohe_raw.index).max()

    genre_counts_ohe = genre_ohe.sum()

    current_genres_to_keep = []
    current_genres_to_sum_into_other = []

    for col_name_full in genre_ohe.columns:
        if genre_counts_ohe[col_name_full] >= min_frequency:
            current_genres_to_keep.append(col_name_full)
        else:
            current_genres_to_sum_into_other.append(col_name_full)

    genre_ohe_working = genre_ohe.copy()
    if current_genres_to_sum_into_other:
        genre_ohe_working['genre_other_genre'] = genre_ohe_working[current_genres_to_sum_into_other].sum(axis=1)
        genre_ohe_working['genre_other_genre'] = (genre_ohe_working['genre_other_genre'] > 0).astype(int)

    final_cols_for_genre_selection = current_genres_to_keep[:]
    if 'genre_other_genre' in genre_ohe_working.columns and 'genre_other_genre' not in final_cols_for_genre_selection:
        final_cols_for_genre_selection.append('genre_other_genre')

    genre_ohe_final = genre_ohe_working[final_cols_for_genre_selection].loc[:, ~genre_ohe_working[final_cols_for_genre_selection].columns.duplicated()]
    genre_ohe_final = genre_ohe_final.astype(int)

    if verbose:
        print(f"[INFO] Processamento de gêneros concluído. Colunas criadas: {genre_ohe_final.shape[1]}")

    return genre_ohe_final
