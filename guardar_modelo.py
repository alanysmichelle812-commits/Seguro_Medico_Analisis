import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler # Necesario para consistencia, aunque RF no lo use directamente

# --- 1. CARGA DE DATOS ---
df_train = pd.read_csv('train.csv')
X_train = df_train.drop('charges', axis=1)
y_train = df_train['charges']

# --- 2. ENTRENAMIENTO DEL MODELO GANADOR (Random Forest) ---
# Usamos los mismos parámetros que nos dieron el mejor resultado
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# --- 3. GUARDAR EL MODELO ---
# Guardar el modelo entrenado en un archivo binario
joblib.dump(rf_model, 'modelo_random_forest.pkl')

print("Modelo Random Forest guardado como 'modelo_random_forest.pkl' para el despliegue web.")