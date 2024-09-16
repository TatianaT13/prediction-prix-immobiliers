from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Charger les données
data = pd.read_csv('data/housing.csv')

# Prétraitement (en supposant que 'median_house_value' est la cible)
X = data.drop(['median_house_value'], axis=1)  # Variables explicatives
y = data['median_house_value']  # Variable cible

# Encodage des variables catégorielles et traitement des valeurs manquantes si nécessaire
X = pd.get_dummies(X, drop_first=True)  # Ex : encodage de 'ocean_proximity'
X.fillna(X.mean(), inplace=True)  # Exemple pour traiter les valeurs manquantes

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
print(f"RMSE : {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R² : {r2_score(y_test, y_pred):.2f}")

# Sauvegarder le modèle pour une utilisation future
import joblib
joblib.dump(model, 'house_price_model.pkl')