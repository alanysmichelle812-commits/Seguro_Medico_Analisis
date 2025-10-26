# Predicción de Costos de Seguros Médicos (Charges)

## Objetivo del Proyecto
El objetivo principal es construir y evaluar un modelo de regresión capaz de predecir el costo individual del seguro médico (`charges`) basándose en factores demográficos y de estilo de vida como la edad, el BMI, el estatus de fumador y la región.

## Fases de Análisis y Preprocesamiento (EDA)

Esta fase se llevó a cabo para limpiar, entender y preparar el dataset para el modelado.

### 1. Limpieza y Filtrado
- **Valores Nulos:** Se confirmó la ausencia de valores nulos.
- **Outliers:** Se eliminaron 9 outliers de la variable **BMI** utilizando el método del Rango Intercuartílico (IQR) para asegurar la robustez del modelo.

### 2. Tratamiento de la Variable Objetivo (`Charges`)
- **Transformación Logarítmica:** La variable `charges` presentaba un fuerte sesgo a la derecha. Se aplicó una transformación logarítmica (log-transform) para acercar la distribución a la normalidad, lo cual es esencial para mejorar el rendimiento de los modelos de regresión lineal.

### 3. Codificación y División
- **Codificación:** Las variables categóricas (`sex`, `smoker`, `region`) fueron codificadas mediante One-Hot Encoding.
- **División:** El dataset preprocesado se dividió en conjuntos de entrenamiento (80%) y prueba (20%) y se guardó en los archivos **`train.csv`** y **`test.csv`**. Se usó estratificación para asegurar una representación proporcional de la variable objetivo en ambos conjuntos.