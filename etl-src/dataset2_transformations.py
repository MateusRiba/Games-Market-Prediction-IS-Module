import pandas as pd

def filter_columns(dataset2):
    cols_to_keep = ['Name', 'Global_Sales', 'JP_Sales', 'NA_Sales', 'PAL_Sales', 'Other_Sales']
    df_filtered = dataset2[cols_to_keep].copy()
    print(f"[INFO] Dataset2 filtrado: mantidas colunas {cols_to_keep} com {len(df_filtered)} linhas.")
    return df_filtered

def group_sales_by_name(dataset2_filtered):
    df_grouped = dataset2_filtered.groupby('Name', as_index=False).sum(numeric_only=True)
    print(f"[INFO] Dataset2 agrupado por 'Name'. Total de jogos após agrupamento: {len(df_grouped)}.")
    return df_grouped

def fill_missing_sales(dataset2_grouped):
    def preencher_sales(row):
        sales_cols = ['JP_Sales', 'NA_Sales', 'PAL_Sales', 'Other_Sales']
        known_sales = [row[col] for col in sales_cols if pd.notna(row[col])]
        missing_cols = [col for col in sales_cols if pd.isna(row[col])]

        if pd.isna(row['Global_Sales']):
            if len(missing_cols) == 0:
                return row
            else:
                row['delete'] = True
                return row

        total_known = sum(known_sales)
        remaining = row['Global_Sales'] - total_known

        if abs(remaining) < 1e-3:
            return row

        if len(missing_cols) > 0:
            distributed_value = remaining / len(missing_cols)
            for col in missing_cols:
                row[col] = distributed_value

        return row

    df = dataset2_grouped.copy()
    df['delete'] = False
    df = df.apply(preencher_sales, axis=1)
    before_drop = len(df)
    df = df[df['delete'] == False].drop(columns=['delete'])
    after_drop = len(df)

    print(f"[INFO] Preenchimento de vendas faltantes finalizado. Removidas {before_drop - after_drop} linhas inválidas.")
    assert df[['JP_Sales', 'NA_Sales', 'PAL_Sales', 'Other_Sales', 'Global_Sales']].isnull().sum().sum() == 0, \
        "[ERRO] Ainda há valores nulos após preenchimento!"
    print(f"[INFO] Todos os valores de vendas estão preenchidos corretamente.")

    return df

def merge_sales_to_dataset1(dataset1, dataset2_filled):
    if 'Name' in dataset2_filled.columns:
        dataset2_filled = dataset2_filled.rename(columns={'Name': 'title'})
    dataset2_filled['title'] = dataset2_filled['title'].str.strip().str.lower()
    dataset1['title'] = dataset1['title'].str.strip().str.lower()

    dataset1_filtered = dataset1[dataset1['title'].isin(dataset2_filled['title'])].copy()
    print(f"[INFO] Dataset1 filtrado para {len(dataset1_filtered)} jogos presentes no Dataset2.")

    merged = pd.merge(
        dataset1_filtered,
        dataset2_filled[['title', 'JP_Sales', 'NA_Sales', 'PAL_Sales', 'Other_Sales']],
        on='title',
        how='left'
    )

    assert merged[['JP_Sales', 'NA_Sales', 'PAL_Sales', 'Other_Sales']].isnull().sum().sum() == 0, \
        "[ERRO] O merge falhou em preencher vendas em alguns jogos!"
    print(f"[INFO] Merge realizado com sucesso. Total de jogos no dataset final: {len(merged)}.")

    return merged

def main_db2(dataset1, dataset2):
    print("[INÍCIO] Iniciando processamento do Dataset2...")
    dataset2_filtered = filter_columns(dataset2)
    dataset2_grouped = group_sales_by_name(dataset2_filtered)
    dataset2_filled = fill_missing_sales(dataset2_grouped)
    final_dataset = merge_sales_to_dataset1(dataset1, dataset2_filled)
    print("[FIM] Processamento concluído com sucesso.")
    return final_dataset
