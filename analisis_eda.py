import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------------
# --- 1. CARGA DE DATOS Y EXPLORACIÓN INICIAL ---
# --------------------------------------------------------------------------
print("--- 1. CARGA DE DATOS Y EXPLORACIÓN INICIAL ---")

# Cargar el DataSet
# Asegúrate de que 'insurance.csv' esté en la misma carpeta del proyecto
df = pd.read_csv('insurance.csv')

# a. Dimensiones y Primeras Filas
print(f"\nDimensiones del Dataset: {df.shape}")
print("\nPrimeras 5 filas (df.head()):")
print(df.head())

# b. Información General (Tipos de Datos y Nulos)
print("\nTipos de datos y conteo de valores no nulos (df.info()):")
df.info()

# c. Estadísticas Descriptivas
print("\nEstadísticas Descriptivas de variables numéricas (df.describe()):")
print(df.describe().T) # La .T transpone la tabla para mejor lectura

print("\nEstadísticas Descriptivas de variables categóricas (df.describe(include='object')):")
print(df.describe(include='object'))

# --------------------------------------------------------------------------
# --- 2. TRATAMIENTO DE VALORES VACÍOS ---
# --------------------------------------------------------------------------
print("\n--- 2. TRATAMIENTO DE VALORES VACÍOS ---")

nulos = df.isnull().sum()
print("Conteo de valores nulos por columna:")
print(nulos[nulos > 0])

# --- EXPLICACIÓN DE POR QUÉ SE HIZO ESTO ---
if nulos.sum() == 0:
    print("\n[EXPLICACIÓN]: No se encontraron valores nulos o faltantes en el dataset. Por lo tanto, no se requirió ninguna acción de imputación o eliminación de filas. Esto garantiza que utilizaremos el 100% de los datos originales.")
else:
    # Si hubiera nulos, aquí iría el código de imputación o eliminación y la justificación.
    print("\n[EXPLICACIÓN]: Se encontraron valores nulos. Se decidió XXXXXX porque XXXXXX.")


# --------------------------------------------------------------------------
# --- 3. TRANSFORMACIONES INICIALES (CORREGIDO: Fuera del bloque 'else') ---
# --------------------------------------------------------------------------
print("\n--- 3. TRANSFORMACIONES INICIALES ---")

# Renombrar columnas para mayor claridad (opcional, si los nombres fueran malos)
# df.rename(columns={'old_name': 'new_name'}, inplace=True) 

# Convertir tipos de datos si fuera necesario (en este dataset, los tipos son correctos)
# Asegurar que 'children' sea vista como categórica para algunos análisis de conteo
df['children'] = df['children'].astype(str)
print("Tipo de 'children' cambiado a 'str' para análisis de conteo.")

# --- EXPLICACIÓN DE POR QUÉ SE HIZO ESTO ---
print("[EXPLICACIÓN]: Las columnas numéricas (age, bmi, charges) y categóricas (sex, smoker, region) tenían los tipos de datos correctos. Solo se convirtió la columna 'children' a tipo string/object para facilitar el análisis univariante y bivariante como una variable categórica de conteo de familias.")

# --- 4. ANÁLISIS UNIVARIANTE ---
print("\n\n--- 4. ANÁLISIS UNIVARIANTE ---")

# ------------------------------
# 4.1. Variables Numéricas
# ------------------------------
print("\n4.1. Análisis de Variables Numéricas (age, bmi, charges):")

numeric_cols = ['age', 'bmi', 'charges']

# Configuración para gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plt.suptitle('Distribución de Variables Numéricas', fontsize=16)

# Generar Histogramas
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(col.capitalize())
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Media')
    axes[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el título
plt.show() # Mostrar los 3 gráficos

# --- DEDUCCIONES NUMÉRICAS ---
print("\n[DEDUCCIONES - NUMÉRICAS]:")

# 1. AGE
print(f"  - AGE (Edad): La distribución es relativamente uniforme. Esto significa que la muestra contiene una representación equitativa de todos los grupos etarios, desde los más jóvenes hasta los más viejos.")

# 2. BMI
print(f"  - BMI (Índice de Masa Corporal): La distribución se asemeja a una campana con una ligera tendencia a la derecha, centrada alrededor de 30. Esto indica que la mayoría de los clientes en la muestra se encuentran en el umbral de sobrepeso/obesidad, lo cual es relevante para el costo del seguro.")

# 3. CHARGES (Variable Objetivo)
print(f"  - CHARGES (Costo del Seguro): La distribución está fuertemente sesgada a la derecha. La mayoría de los clientes pagan costos bajos (< $10,000), pero hay una larga cola de costos muy altos. Esto confirma la necesidad de transformar esta variable antes del modelado.")
# --- 5. FILTRADO DE VARIABLES Y TRANSFORMACIÓN DEL TARGET ---
print("\n\n--- 5. FILTRADO DE VARIABLES Y TRANSFORMACIÓN DEL TARGET ---")

# ----------------------------------------------------
# 5.1. ELIMINACIÓN DE OUTLIERS EN VARIABLES PREDICTORAS (BMI)
# ----------------------------------------------------
# Aplicaremos la eliminación de outliers al BMI, ya que 'age' está uniformemente distribuida
# y 'charges' es la variable objetivo (la trataremos con logaritmo, no eliminación).

print("\n5.1. Eliminación de Outliers en BMI (Método IQR)")
columna_outlier = 'bmi'

# Calcular Q1, Q3 y el Rango Intercuartílico (IQR)
Q1 = df[columna_outlier].quantile(0.25)
Q3 = df[columna_outlier].quantile(0.75)
IQR = Q3 - Q1

# Definir límites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Contar cuántos outliers hay
outliers_count = df[(df[columna_outlier] < limite_inferior) | (df[columna_outlier] > limite_superior)].shape[0]

# Filtrar el DataFrame
df_filtrado = df[(df[columna_outlier] >= limite_inferior) & (df[columna_outlier] <= limite_superior)].copy()

print(f"  - Outliers detectados en {columna_outlier}: {outliers_count} registros.")
print(f"  - Registros antes del filtro: {df.shape[0]}")
print(f"  - Registros después del filtro: {df_filtrado.shape[0]}")

# Actualizar el DataFrame principal (opcional, pero recomendado para el resto del análisis)
df = df_filtrado.copy()

# --- EXPLICACIÓN DE POR QUÉ SE REALIZA ESTE CORTE ---
print("\n[EXPLICACIÓN FILTRADO]: Se realizó un corte por el Rango Intercuartílico (IQR) en la columna BMI. Esto se debe a que los valores extremos de BMI (outliers) podrían sesgar el entrenamiento del modelo de regresión, haciendo que los coeficientes de predicción se ajusten incorrectamente. Al eliminar estos valores atípicos, garantizamos un modelo más robusto que generalice mejor a la mayoría de la población.")


# ----------------------------------------------------
# 5.2. TRATAMIENTO DE LA VARIABLE OBJETIVO (CHARGES)
# ----------------------------------------------------
print("\n5.2. Tratamiento de la Variable Objetivo (Charges)")

# Aplicar la transformación logarítmica
df['charges_log'] = np.log(df['charges'])

# Graficar para visualizar el efecto de la transformación
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['charges'], kde=True)
plt.title('Charges (Original - Sesgado a la Derecha)')

plt.subplot(1, 2, 2)
sns.histplot(df['charges_log'], kde=True)
plt.title('Charges (Log-Transformado - Casi Normal)')

plt.tight_layout()
plt.show()

# --- EXPLICACIÓN DE POR QUÉ SE REALIZA ESTA TRANSFORMACIÓN ---
print("\n[EXPLICACIÓN TRATAMIENTO TARGET]: Se aplicó una transformación logarítmica (np.log) a la variable objetivo 'Charges'. Según el análisis univariante (histograms), esta variable estaba fuertemente sesgada a la derecha. La transformación logarítmica es necesaria en modelos de regresión para: \n  1. Estabilizar la varianza. \n  2. Acercar la distribución a la normalidad, mejorando el rendimiento de los modelos lineales.")

# Eliminamos la columna original de charges para usar solo la transformada en el resto del análisis
df.drop('charges', axis=1, inplace=True)
df.rename(columns={'charges_log': 'charges'}, inplace=True)
print("\nVariable 'charges' original eliminada y reemplazada por 'charges_log'.")
# --- 6. ANÁLISIS BIVARIANTE Y MULTIVARIANTE ---
print("\n\n--- 6. ANÁLISIS BIVARIANTE Y MULTIVARIANTE ---")

# ----------------------------------------------------
# 6.1. Relación de la Variable Objetivo (Charges Log) vs. Insumos
# ----------------------------------------------------
print("\n6.1. Análisis Bivariante (Charges Log vs. Variables):")

# Variables Categóricas
categorical_cols = ['sex', 'smoker', 'region', 'children'] 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
plt.suptitle('Charges (Log) vs. Variables Categóricas', fontsize=16)

for i, col in enumerate(categorical_cols):
    sns.boxplot(x=col, y='charges', data=df, ax=axes[i], palette='Pastel1')
    axes[i].set_title(f'Costo Log vs. {col.capitalize()}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show() # Mostrar Boxplots

# Variables Numéricas
plt.figure(figsize=(12, 5))
plt.suptitle('Charges (Log) vs. Variables Numéricas', fontsize=16)

plt.subplot(1, 2, 1)
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Charges Log vs. Age')

plt.subplot(1, 2, 2)
sns.scatterplot(x='bmi', y='charges', data=df)
plt.title('Charges Log vs. BMI')

plt.tight_layout()
plt.show() # Mostrar Scatterplots
print("\n[EXPLICACIÓN RELACIONES]:")

# 1. SMOKER
print("  - SMOKER: Es la variable con mayor impacto (relación más fuerte). El boxplot muestra que los fumadores (yes) tienen un costo promedio y una variabilidad significativamente más alta que los no fumadores (no).")

# 2. AGE
print("  - AGE: El scatterplot muestra una clara relación lineal positiva. A medida que la edad aumenta, el costo del seguro (log) también aumenta de manera predecible. Esto confirma que la edad es un predictor clave.")

# 3. BMI
print("  - BMI: Existe una relación positiva, aunque menos pronunciada que la edad. Las personas con mayor BMI tienden a tener mayores costos, especialmente en el grupo de obesos.")

# 4. REGION, SEX, CHILDREN
print("  - REGION / SEX / CHILDREN: Estas variables muestran poca variación en el costo promedio. Su impacto es menor y es probable que solo modifiquen marginalmente las predicciones.")
# ----------------------------------------------------
# 6.3. Matriz de Correlación y Multicolinealidad
# ----------------------------------------------------
print("\n6.3. Matriz de Correlación (Evaluación de Multicolinealidad):")

# A. Codificación para la Matriz
# One-Hot Encoding para las nominales
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Convertir 'children' de nuevo a numérico si es necesario (ya que fue string en EDA)
df_encoded['children'] = pd.to_numeric(df_encoded['children'])

# B. Cálculo de la Matriz de Correlación
corr_matrix = df_encoded.corr()

# C. Visualización de la Matriz (Heatmap)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación del Dataset Codificado')
plt.show()

# D. Análisis de Multicolinealidad (entre variables NO objetivo)
# Revisar correlaciones entre variables de insumo. Buscamos valores cercanos a 1 o -1.

print("\n[ANÁLISIS DE MULTICOLINEALIDAD]:")

# Extraer correlaciones entre predictores (excluyendo el target 'charges')
predictor_corr = corr_matrix.drop('charges', axis=0).drop('charges', axis=1)

# Obtener los valores absolutos de correlación fuera de la diagonal
predictor_corr_abs = predictor_corr.abs()
np.fill_diagonal(predictor_corr_abs.values, 0) # Ignorar la diagonal (correlación consigo mismo = 1)

# Encontrar el par con la correlación más alta
max_corr = predictor_corr_abs.unstack().sort_values(ascending=False)
max_pair = max_corr[max_corr < 1.0].head(1)

if max_pair.empty or max_pair.iloc[0] < 0.8: # Umbral común para multicolinealidad
    print("  - No se encontraron correlaciones entre variables de insumo lo suficientemente altas (r > 0.8) para justificar la eliminación de una variable por multicolinealidad.")
    print("  - La correlación más alta entre predictores es:", max_pair)
    # No eliminamos ninguna columna ya que no hay alta multicolinealidad.
else:
    # Si hubiera una correlación muy alta (ej: Age y Years_Employed)
    col1, col2 = max_pair.index[0]
    print(f"  - Correlación más alta detectada entre {col1} y {col2}: {max_pair.iloc[0]:.2f}")
    # Decisión de eliminación y explicación

# --- EXPLICACIÓN DE LA DECISIÓN ---
print("\n[EXPLICACIÓN DE LA MATRIZ Y ELIMINACIÓN]:")
print("La matriz de correlación visualizada confirma que la variable más fuertemente correlacionada con el costo ('charges') es 'smoker_yes'.")
print("No se encontró una alta multicolinealidad (correlación |r| > 0.8) entre las variables de insumo (predictoras) restantes. Por lo tanto, se conservarán todas las variables de insumo, ya que cada una aporta información única al modelo de regresión.")

# Guardamos el DataFrame codificado para la división
df = df_encoded.copy()
# --- 7. DIVISIÓN DEL DATASET (80/20) ---
print("\n\n--- 7. DIVISIÓN DEL DATASET (80/20) ---")

from sklearn.model_selection import train_test_split

# Variables independientes (X) y la variable objetivo (y)
X = df.drop('charges', axis=1)
y = df['charges']

# 1. Creación de Bines para Estratificación (Recomendado para Regresión)
# Crear bines basados en la variable objetivo 'charges'
y_binned = pd.cut(y, bins=5, labels=False, include_lowest=True)

# 2. División con Estratificación
# Aplicamos la estratificación sobre los bines creados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    shuffle=True, 
    stratify=y_binned # Estratificar usando los bines
)

# Recombinar para guardar los archivos train.csv y test.csv
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# Guardar los DataFrames
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

print("\nDataset dividido y guardado:")
print(f"  - train.csv guardado: {df_train.shape[0]} registros (80%)")
print(f"  - test.csv guardado: {df_test.shape[0]} registros (20%)")


# 3. Comprobación de la Proporción (Estratificación)
print("\nComprobación de Proporciones (Usando Bines):")
original_prop = y_binned.value_counts(normalize=True).sort_index()
train_prop = pd.cut(y_train, bins=5, labels=False, include_lowest=True).value_counts(normalize=True).sort_index()
test_prop = pd.cut(y_test, bins=5, labels=False, include_lowest=True).value_counts(normalize=True).sort_index()

proportions = pd.DataFrame({
    'Original': original_prop,
    'Train': train_prop,
    'Test': test_prop
})

print(proportions)
print("\n[EXPLICACIÓN DIVISIÓN]: El dataset se dividió usando una regla 80/20. Se aplicó estratificación sobre bines de la variable objetivo 'charges' para asegurar que la proporción de costos (bajos, medios, altos) sea mantenida de manera similar en los conjuntos de entrenamiento y prueba.") 