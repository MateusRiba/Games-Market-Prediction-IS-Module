import pandas as pd

def read_data(csv_file1):
    try:
        dataset = pd.read_csv(csv_file1)
    except FileNotFoundError:
        print(f"O arquivo {csv_file1} não foi encontrado.")
        return None
    except pd.errors.EmptyDataError:
        print(f"O arquivo {csv_file1} está vazio.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao acessar o arquivo: {e}")
        return None
    else:
        print(f"Dataset {csv_file1} carregado com sucesso!")
        return dataset
