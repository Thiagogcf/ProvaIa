import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset
data = pd.read_csv("gdp_csv.csv")

# Visualizando as primeiras linhas do dataset
print(data.head())

# Verificando se há dados faltantes
print(data.isnull().sum())

# Removendo dados faltantes
data.dropna(inplace=True)

# Gráficos exploratórios
numeric_data = data[['Year', 'Value']]

sns.pairplot(numeric_data)
plt.show()

sns.heatmap(numeric_data.corr(method='pearson'), annot=True)
plt.show()

sns.boxplot(data=numeric_data)
plt.show()

sns.histplot(numeric_data)
plt.show()

# Separando as variáveis independentes e a variável dependente
X = data.drop(['Value', 'Country Name', 'Country Code'], axis=1)
y = data['Value']

# Dividindo o dataset em conjunto de treinamento e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear Múltipla
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# KNN
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# MLP
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# Comparando os resultados
print("MSE Linear:", mse_linear)
print("R2 Linear:", r2_linear)

print("MSE KNN:", mse_knn)
print("R2 KNN:", r2_knn)

print("MSE MLP:", mse_mlp)
print("R2 MLP:", r2_mlp)
