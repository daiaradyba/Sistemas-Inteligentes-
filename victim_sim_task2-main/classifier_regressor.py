import numpy as np
import pandas as pd
import joblib
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os

PARAM_DT = {
    'criterion': ['squared_error'],
    'splitter': ['best'],
    'max_depth': [16],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    #'max_features': [None, 'auto', 'sqrt', 'log2'],
    'random_state': [42],
    'max_leaf_nodes': [100, 200],
    #'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
}
# Limitar a quantidade max de nós folha pode ajudar a evitar overfitting

PARAM_MLP_1 = {
    'hidden_layer_sizes': [(100, 100, 100, 100)],
    'max_iter': [3000],
    'activation': ['tanh'],
    'solver': ['adam'],        # 'sgd' não funciona com learning_rate = 'adaptive', e funciona melhor com dataset grande
    'alpha': [0.0001],     # default = 0.0001, L2 regularization
    'random_state': [42]
}

def load_csv(file_path):
    # Carregue o arquivo CSV
    aux = pd.read_csv(file_path, header=None)
    data = aux.iloc[:, [3, 4, 5, 6]]
    print(data)
    return data

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)

    clf = GridSearchCV(model, PARAM_DT, cv=3)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Best model MSE:", mse)

    # Se a acurácia do treinamento for maior que a do teste, pode estar ocorrendo overfitting
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    return clf, best_model, mse, train_accuracy, test_accuracy

def train_mlp(X_train, X_test, y_train, y_test):    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Definir o modelo MLP
    mlp = MLPRegressor(random_state=42)
    
    # Usar GridSearchCV para encontrar os melhores hiperparâmetros
    clf = GridSearchCV(mlp, PARAM_MLP_1, verbose=1, cv=3)
    clf.fit(X_train, y_train)
    
    best_model = clf.best_estimator_
    
    # Avaliar o modelo no conjunto de teste
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MLP Model MSE:", mse)
    
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    return clf, best_model, mse, train_accuracy, test_accuracy, scaler

def main():
    FILE_PATH = r"datasets\data_4000v\env_vital_signals.txt"
    data = load_csv(FILE_PATH)
    X = data.iloc[:, :-1]  # Variáveis independentes
    y = data.iloc[:, -1]   # Variável dependente

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

    # Treinamento do modelo da árvore de decisão
    dt_clf, best_model, mse, train_accuracy, test_accuracy = train_decision_tree(X_train, X_test, y_train, y_test)
    print(f"Best Decision Tree Model: {best_model}")
    print(f"MSE: {mse}")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    print("\n")
    print("###########################################")
    print("\n")

    start_time = time.time()

    # Treinamento do modelo MLP
    mlp_clf, best_mlp_model, mlp_mse, mlp_train_accuracy, mlp_test_accuracy, scaler = train_mlp(X_train, X_test, y_train, y_test)
    print(f"Best MLP Model: {best_mlp_model}")
    print(f"MSE: {mlp_mse}")
    print(f"Training Accuracy: {mlp_train_accuracy}")
    print(f"Test Accuracy: {mlp_test_accuracy}")

    end_time = time.time()

    # Avaliar os modelos com o dataset de 800 vítimas
    FILE_PATH_TEST = r"datasets\data_408v_94x94\env_vital_signals_teste_cego.txt"
    data_test = load_csv(FILE_PATH_TEST)
    X_test = data_test.iloc[:, :-1]
    y_test = data_test.iloc[:, -1]

    # Avaliar o modelo da árvore de decisão
    y_pred = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)

    print("Decision Tree Model MSE Test:", mse_test)
    print("Decision Tree Model Accuracy Test:", best_model.score(X_test, y_test))

    # Avaliar o modelo MLP
    X_test_mlp = scaler.transform(X_test)
    y_pred = best_mlp_model.predict(X_test_mlp)
    mlp_mse_test = mean_squared_error(y_test, y_pred)

    print("MLP Model MSE Test:", mlp_mse_test)
    print("MLP Model Accuracy Test:", best_mlp_model.score(X_test_mlp, y_test))

    # Salvar os modelos
    # Criar uma pasta com a data e horário de criação
    folder_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    folder_path = os.path.join("models", folder_name)
    os.makedirs(folder_path)

    # Mover os modelos para a nova pasta
    decision_tree_model_path = os.path.join(folder_path, "decision_tree_model.pkl")
    mlp_model_path = os.path.join(folder_path, "mlp_model.pkl")
    scaler_path = os.path.join(folder_path, "scaler.pkl")
    joblib.dump(best_model, decision_tree_model_path)
    joblib.dump(best_mlp_model, mlp_model_path)
    joblib.dump(scaler, scaler_path)

    # Atualizar o caminho do arquivo de informações do modelo
    model_info_path = os.path.join(folder_path, "model_info.txt")

    with open(model_info_path, "w") as file:
        # Salvar os parâmetros usados em cada modelo
        file.write("Decision Tree Parameters:\n")
        file.write(json.dumps(dt_clf.best_params_))
        file.write("\n\n")
        file.write("MLP Parameters:\n")
        file.write(json.dumps(mlp_clf.best_params_))
        file.write("\n\n")

        file.write("Best Decision Tree Model:\n")
        file.write(str(best_model))
        file.write("\n\n")
        file.write("MSE: ")
        file.write(str(mse))
        file.write("\n")
        file.write("Training Accuracy: ")
        file.write(str(train_accuracy))
        file.write("\n")
        file.write("Test Accuracy: ")
        file.write(str(test_accuracy))
        file.write("\n\n")

        file.write("Best MLP Model:\n")
        file.write(str(best_mlp_model))
        file.write("\n\n")
        file.write("MSE: ")
        file.write(str(mlp_mse))
        file.write("\n")
        file.write("Training Accuracy: ")
        file.write(str(mlp_train_accuracy))
        file.write("\n")
        file.write("Test Accuracy: ")
        file.write(str(mlp_test_accuracy))
        file.write("\n\n")

        file.write("Using 800v dataset:\n")
        file.write("Decision Tree Model MSE Test: ")
        file.write(str(mse_test))
        file.write("\n")
        file.write("Decision Tree Model Accuracy Test: ")
        file.write(str(best_model.score(X_test, y_test)))
        file.write("\n")
        file.write("MLP Model MSE Test: ")
        file.write(str(mlp_mse_test))
        file.write("\n")
        file.write("MLP Model Accuracy Test: ")
        file.write(str(best_mlp_model.score(X_test_mlp, y_test)))
        # Calcular o tempo que levou para gerar os modelos
        execution_time = end_time - start_time
        file.write("\n\n")
        file.write("Execution Time: ")
        file.write(str(execution_time))
        file.write(" seconds")
    
    # Salvar os parâmetros em arquivos separados
    with open(os.path.join(folder_path, "PARAM_DT.json"), "w") as file:
        json.dump(PARAM_DT, file)

    with open(os.path.join(folder_path, "PARAM_MLP.json"), "w") as file:
        json.dump(PARAM_MLP_1, file)

if __name__ == "__main__":
    main()