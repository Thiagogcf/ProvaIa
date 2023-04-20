import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Carregar o conjunto de dados (substitua o caminho do arquivo pelo caminho do seu arquivo CSV)
data = pd.read_csv("gdp_csv.csv")

# Verificar valores ausentes
print(data.isnull().sum())

# Preencher valores ausentes com a média, se necessário
data.fillna(data.mean(numeric_only=True), inplace=True)

# Converter as colunas 'Country Name' e 'Country Code' para valores numéricos
encoder = LabelEncoder()
data['Country Name'] = encoder.fit_transform(data['Country Name'])
data['Country Code'] = encoder.fit_transform(data['Country Code'])

# Gráfico de correlação
sns.heatmap(data.corr(), annot=True)
plt.show()

# Box Plot
sns.boxplot(data=data)
plt.show()

# Gráfico de frequência
data.hist()
plt.show()

# Definir variáveis preditoras (X) e a variável alvo (y)
X = data.drop(["Value"], axis=1)
y = data["Value"]

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear Múltipla
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)

# KNN
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)

# Métricas de avaliação
print("Regressão Linear Múltipla: MSE =", mean_squared_error(y_test, lr_predictions), "R2 =", r2_score(y_test, lr_predictions))
print("KNN: MSE =", mean_squared_error(y_test, knn_predictions), "R2 =", r2_score(y_test, knn_predictions))
print("MLP: MSE =", mean_squared_error(y_test, mlp_predictions), "R2 =", r2_score(y_test, mlp_predictions))
