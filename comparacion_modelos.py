import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# --- 1. CARGA DE DATOS ---
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.drop('charges', axis=1)
y_train = df_train['charges']
X_test = df_test.drop('charges', axis=1)
y_test = df_test['charges']

# --- 2. DEFINICIÓN DE MODELOS ---
models = {
    'Regresion Lineal': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

resultados_comparacion = {}

print("Iniciando entrenamiento y evaluación de modelos...")

# --- 3. BUCLE DE ENTRENAMIENTO Y EVALUACIÓN ---
for nombre, modelo in models.items():
    start_time = time.time()
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # Predecir sobre el conjunto de prueba
    y_pred_log = modelo.predict(X_test)
    
    # Deshacer la transformación Logarítmica para obtener métricas reales (en dólares)
    y_pred_original = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)
    
    # Calcular métricas
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    end_time = time.time()
    
    resultados_comparacion[nombre] = {
        'R2': r2,
        'RMSE': rmse,
        'Tiempo (s)': end_time - start_time
    }
    
    print(f"[{nombre}] - R2: {r2:.4f}, RMSE: ${rmse:,.2f}")

# --- 4. COMPARACIÓN DE MÉTRICAS ---
df_resultados = pd.DataFrame(resultados_comparacion).T.sort_values(by='R2', ascending=False)

print("\n\n--- TABLA FINAL DE COMPARACIÓN DE MODELOS ---")
print(df_resultados)

# --- 5. EXTRACCIÓN DE IMPORTANCIA DE VARIABLES (SOLO PARA MODELOS DE ENSAMBLE) ---
print("\n--- IMPORTANCIA DE VARIABLES (Random Forest) ---")
rf_model = models['Random Forest']
importancias = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(importancias)