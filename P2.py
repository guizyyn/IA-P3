#importando as bibliotecas

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#carregando o conjunto de dados
california = fetch_california_housing()

#variaveis independentes (features)
x = california.data

#variaveis dependetes (target)
y = california.target

#dividimos os dados para o treinamento e para os testes
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.3, random_state=42)

#criando o modelo de regressão linear
model = LinearRegression()

#treinando o modelo
model.fit(x_train, y_train)

#previções com os dados de teste
y_pred = model.predict(x_test)

#avaliando as metricas para determinar o desempenho
mae = mean_absolute_error(y_test, y_pred)  #erro medio absoluto
mse = mean_squared_error(y_test, y_pred)   #erro quadratico medio
rmse = np.sqrt(mse)                        #raiz quadrada do erro quadratico medio
r2 = r2_score(y_test, y_pred)              #coificiente de determinação

print = (f"ERRO MEDIO ABSOLUTO (MAE): {mae}")
print = (f"ERRO QUADRATICO MEDIO (MSE): {mse}")
print = (f"RAIZ DO ERRO QUADRATICO MEDIO (RMSE): {rmse}")
print = (f"R² SCORE: {r2}")

#plotando para vizualizarmos os resultados
plt.scatter(x_test[:, 0], y_test, color = 'blue', label = 'dados reais 00')
plt.scatter(x_test[:, 0], y_pred, color = 'red', label = 'previsões 00', alpha = 0.7)
plt.scatter(x_test[:, 1], y_test, color = 'blue', label = 'dados reais 01')
plt.scatter(x_test[:, 1], y_pred, color = 'black', label = 'previsões 01', alpha = 0.7)
plt.scatter(x_test[:, 2], y_test, color = 'gray', label = 'dados reais 02')
plt.scatter(x_test[:, 2], y_pred, color = 'violet', label = 'previsões 02', alpha = 0.7)
plt.scatter(x_test[:, 3], y_test, color = 'red', label = 'dados reais 03')
plt.scatter(x_test[:, 3], y_pred, color = 'purple', label = 'previsões 03', alpha = 0.7)
plt.scatter(x_test[:, 4], y_test, color = 'orange', label = 'dados reais 04')
plt.scatter(x_test[:, 4], y_pred, color = 'brown', label = 'previsões 04', alpha = 0.7)
plt.scatter(x_test[:, 5], y_test, color = 'yellow', label = 'dados reais 05')
plt.scatter(x_test[:, 5], y_pred, color = 'black', label = 'previsões 05', alpha = 0.7)
plt.xlabel(california.feature_names[0])
plt.xlabel(california.feature_names[1])
plt.xlabel(california.feature_names[2])
plt.xlabel(california.feature_names[3])
plt.xlabel(california.feature_names[4])
plt.xlabel(california.feature_names[5])
plt.ylabel('mediana dos valores das casas')
plt.title('REGRESSÃO LINEAR - CALIFORNIA HOUSING')
plt.legend()
plt.show()
