# Proyecto de Machine Learning: Predicción de Costos de Seguros Médicos

##  Objetivo del Proyecto
El objetivo principal es construir, evaluar y optimizar un modelo de regresión capaz de predecir el gasto médico individual (`charges`) con alta precisión.

##  Fases de Análisis y Preprocesamiento (EDA)

Esta fase se llevó a cabo para limpiar, entender y preparar el dataset para el modelado:

1.  **Limpieza de Outliers:** Se eliminaron 9 outliers de la variable **BMI** (usando el método IQR) para evitar sesgos en el entrenamiento.
2.  **Transformación del Target:** La variable objetivo (`charges`) fue transformada logarítmicamente (`np.log()`) debido a su fuerte sesgo a la derecha, mejorando la distribución para el modelado.
3.  **Codificación y División:** Las variables categóricas fueron codificadas (One-Hot Encoding). El dataset preprocesado se dividió en conjuntos de entrenamiento y prueba (80/20).

##  Resultados del Modelado

Se evaluaron tres modelos de regresión para seleccionar el de mejor rendimiento. Las métricas se calcularon revirtiendo la transformación logarítmica para obtener valores en dólares.

| Modelo | $R^2$ (Explicación de Varianza) | RMSE (Error Promedio) |
| :--- | :--- | :--- |
| **Random Forest Regressor** | **0.8574** | **$4,453.99** |
| Gradient Boosting Regressor | 0.8570 | $4,459.65 |
| Regresión Lineal | 0.5648 | $7,779.61 |

**Modelo Ganador:** El **Random Forest Regressor** fue elegido como el modelo final, ya que logró aumentar el poder predictivo a **85.74%** y redujo significativamente el error promedio.

##  Conclusiones 

El análisis de la **Importancia de Variables** del modelo Random Forest reveló que los costos del seguro están impulsados principalmente por:

1.  **Edad (37.1%)**: El factor individual más influyente.
2.  **Estado de Fumador (44% total)**: El factor de riesgo más significativo.
3.  **BMI (10.1%)**.

El modelo final es robusto y ofrece una base sólida para la estimación de gastos médicos.
