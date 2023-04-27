import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Carregar o conjunto de dados em um dataframe do Pandas
df = pd.read_csv('gdp_csv.csv')

# Visualizar as primeiras linhas do dataframe
print(df.head())

# Verificar se há valores nulos ou faltantes
print(df.isnull().sum())

# Converter colunas categóricas em numéricas usando LabelEncoder
label_encoder = LabelEncoder()
df['Country Name'] = label_encoder.fit_transform(df['Country Name'])
df['Country Code'] = label_encoder.fit_transform(df['Country Code'])

# Explorar os dados por meio de gráficos
sns.pairplot(df)
sns.heatmap(df.corr(), annot=True)
plt.show()

# Separar os dados em conjunto de treinamento e conjunto de testes
X = df[['Country Name', 'Country Code', 'Year']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regressão Linear Múltipla
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Regressão Linear Múltipla')
print('R2 Score:', r2_score(y_test, y_pred_lr))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# KNN
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('KNN')
print('R2 Score:', r2_score(y_test, y_pred_knn))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_knn)))

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print('MLP')
print('R2 Score:', r2_score(y_test, y_pred_mlp))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_mlp)))
