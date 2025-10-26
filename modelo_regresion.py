import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # El modelo que usaremos primero
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- 1. CARGA DE DATOS ---
print("Cargando datasets de entrenamiento y prueba...")
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# --- 2. DEFINICIÓN DE VARIABLES (X e y) ---
# Separar características (X) y la variable objetivo (y)
X_train = df_train.drop('charges', axis=1)
y_train = df_train['charges']

X_test = df_test.drop('charges', axis=1)
y_test = df_test['charges']

# --- 3. ENTRENAMIENTO DEL MODELO ---
print("Entrenando Modelo de Regresión Lineal...")
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

# --- 4. PREDICCIÓN ---
y_pred_log = modelo_rl.predict(X_test)

# --- 5. EVALUACIÓN Y DESHACER LA TRANSFORMACIÓN LOGARÍTMICA ---

# Convertir las predicciones de vuelta a la escala original (dólares)
y_pred_original = np.exp(y_pred_log)
y_test_original = np.exp(y_test)

# Calcular métricas con los valores originales
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print("\n--- Resultados de la Evaluación ---")
print(f"Error Cuadrático Medio (MSE): {mse:,.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:,.2f}")
print(f"Coeficiente de Determinación (R-cuadrado): {r2:.4f}")

# Opcional: Mostrar una predicción de ejemplo
print(f"\nPredicción de ejemplo (Costo real vs. Predicción):")
for i in range(5):
    print(f"  Real: ${y_test_original.iloc[i]:,.2f} | Predicción: ${y_pred_original[i]:,.2f}")
    