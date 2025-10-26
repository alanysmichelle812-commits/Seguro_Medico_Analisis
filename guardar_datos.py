import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. CARGA DE DATOS (Mismo inicio que analisis_eda.py) ---
df = pd.read_csv('insurance.csv')

# --- 2. PREPROCESAMIENTO Y LIMPIEZA (Secciones 3 y 5) ---
print("Iniciando Preprocesamiento y Limpieza...")

# 2.1. Transformación de 'children' a string para compatibilidad inicial (Sección 3)
df['children'] = df['children'].astype(str)

# 2.2. Eliminación de Outliers en BMI (Sección 5.1)
columna_outlier = 'bmi'
Q1 = df[columna_outlier].quantile(0.25)
Q3 = df[columna_outlier].quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
df_filtrado = df[(df[columna_outlier] >= limite_inferior) & (df[columna_outlier] <= limite_superior)].copy()
df = df_filtrado.copy()
print(f"  - Outliers de BMI eliminados. Registros restantes: {df.shape[0]}")

# 2.3. Tratamiento Logarítmico del Target (Sección 5.2)
df['charges_log'] = np.log(df['charges'])
df.drop('charges', axis=1, inplace=True)
df.rename(columns={'charges_log': 'charges'}, inplace=True)
print("  - Variable Target (charges) transformada logarítmicamente.")


# --- 3. CODIFICACIÓN (Sección 6.3) ---
# One-Hot Encoding para todas las categóricas
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
df = df_encoded.copy()

# 'children' era string y ahora debe ser numérica para el modelo
df['children'] = pd.to_numeric(df['children'])
print("  - Variables categóricas codificadas (One-Hot).")

# --- 4. DIVISIÓN Y GUARDADO (Sección 7) ---
print("\nIniciando División y Guardado...")

X = df.drop('charges', axis=1)
y = df['charges']

# Creación de Bines para Estratificación
y_binned = pd.cut(y, bins=5, labels=False, include_lowest=True)

# División con Estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True, 
    stratify=y_binned 
)

# Recombinar y guardar
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

print("\n--- ¡TAREA COMPLETADA! ---")
print(f"Dataset dividido y guardado:")
print(f"  - train.csv: {df_train.shape[0]} registros")
print(f"  - test.csv: {df_test.shape[0]} registros")
