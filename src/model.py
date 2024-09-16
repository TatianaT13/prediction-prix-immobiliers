import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('median_house_value')
    data = data[numerical_features + ['median_house_value']].dropna()
    X = data[numerical_features]
    y = data['median_house_value']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}")

def main():
    data = load_data('data/housing.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()