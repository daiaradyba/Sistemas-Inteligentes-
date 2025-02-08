import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_csv(file_path):
    aux = pd.read_csv(file_path, header=None)
    data = aux.iloc[:, [3, 4, 5, 6]]
    return data

def load_csv_target(file_path):
    aux = pd.read_csv(file_path, header=None)
    data = aux.iloc[:, [1, 2, 3]]
    return data

def main():
    # Caminhos dos arquivos
    model_path = r"models\mlp_model.pkl"
    scaler_path = r"models\scaler.pkl"
    train_file_path = r"datasets\data_4000v\env_vital_signals.txt"
    test_file_path = r"datasets\data_408v_94x94\env_vital_signals_cego.txt"
    target_file_path = r"datasets\data_408v_94x94\target.txt"

    # Carregar o modelo e o scaler
    mlp_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Carregar os dados de treinamento
    train_data = load_csv(train_file_path)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # Carregar os dados de teste
    test_data = load_csv(test_file_path)
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Comparar com target.txt
    target = load_csv_target(target_file_path)
    y_target = target.iloc[:, -1]
    print(y_target)

    # Normalizar os dados
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fazer previsões no conjunto de teste
    y_pred_train = mlp_model.predict(X_train)
    y_pred_test = mlp_model.predict(X_test)

    # Calcular MSE
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_target, y_pred_test)

    # Calcular acurácia
    train_accuracy = mlp_model.score(X_train, y_train)
    test_accuracy = mlp_model.score(X_test, y_target)

    rmse_target = root_mean_squared_error(y_target, y_pred_test)

    # Imprimir os resultados
    print(f"MLP Model MSE (Target): {mse_test}")
    print(f"MLP Model RMSE (Target): {rmse_target}")
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()