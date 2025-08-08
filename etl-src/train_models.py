import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_and_evaluate_model(dataset, target_column):
    print(f"\n--- Treinando modelo para a variável alvo: '{target_column}' ---")

    dataset = dataset[dataset[target_column].notna()].copy()
    print(f"[INFO] Quantidade de amostras após remover NaNs no target '{target_column}': {len(dataset)}")

    Y = dataset[target_column]
    target_variables_all = ['metascore', 'NA_Sales', 'PAL_Sales', 'JP_Sales', 'Other_Sales']
    X_cols_to_drop = [col for col in target_variables_all if col != target_column]
    X = dataset.drop(columns=X_cols_to_drop + [target_column], errors='ignore')

    numerical_cols_with_nan = X.select_dtypes(include=np.number).columns[X.select_dtypes(include=np.number).isnull().any()].tolist()
    if numerical_cols_with_nan:
        imputer = SimpleImputer(strategy='mean')
        X[numerical_cols_with_nan] = imputer.fit_transform(X[numerical_cols_with_nan])
        print(f"[INFO] Imputados NaNs nas colunas numéricas: {numerical_cols_with_nan}")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    print(f"[INFO] Dados divididos em treino ({len(X_train)}) e teste ({len(X_test)})")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[INFO] Modelo treinado.")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    print("\nExemplo de 10 previsões vs valores reais:")
    comparison_df = pd.DataFrame({
        'Valor Real': y_test.values[:10],
        'Previsão': y_pred[:10]
    })
    print(comparison_df)

    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.join(models_dir, f'model_{target_column}.joblib')
    joblib.dump(model, model_filename)
    print(f"[INFO] Modelo salvo em '{model_filename}'")

    # SALVAR FEATURES
    feature_names = X_train.columns.tolist()
    features_filename = os.path.join(models_dir, f'features_{target_column}.pkl')
    joblib.dump(feature_names, features_filename)
    print(f"[INFO] Colunas de features salvas em '{features_filename}'")

    return model, X_test
