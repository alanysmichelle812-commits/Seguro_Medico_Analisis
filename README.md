# Proyecto de Machine Learning: Predicción de Costos de Seguros Médicos

## Aplicación Web Desplegada
Puedes interactuar con el modelo de predicción de costos directamente aquí:

[**IR A LA APLICACIÓN WEB**](https://xoqkybvbyltjtsfpr3qqyf.streamlit.app/)

---

## Objetivo del Proyecto
El objetivo principal es construir, evaluar y optimizar un modelo de regresión capaz de predecir el gasto médico individual ("charges") con alta precisión, identificando los factores de riesgo más influyentes.

---

##  Fases de Análisis y Preprocesamiento (EDA)

Esta fase se llevó a cabo para limpiar, entender y preparar el dataset para el modelado:

1.  **Limpieza de Outliers:** Se eliminaron 9 *outliers* de la variable **BMI** (usando el método IQR) para evitar sesgos en el entrenamiento.
2.  **Transformación del Target:** La variable objetivo (`charges`) fue transformada **logarítmicamente** (`np.log()`) debido a su fuerte sesgo a la derecha, mejorando la distribución para el modelado.
3.  **Codificación y División:** Las variables categóricas fueron codificadas (**One-Hot Encoding**). El dataset preprocesado se dividió en conjuntos de entrenamiento y prueba (80/20).

---

## Resultados del Modelado y Comparación

Se evaluaron **cuatro modelos de regresión** (incluyendo el modelo base de Regresión Lineal) para seleccionar el de mejor rendimiento. Las métricas se calcularon revirtiendo la transformación logarítmica para obtener valores en dólares.

| Modelo | $R^2$ (Varianza Explicada) | RMSE (Error Promedio) | MAPE (Error Porcentual) |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **0.8574** | **$4,453.99** | 20.08% |
| Gradient Boosting Regressor | 0.8570 | $4,459.65 | **16.42%** |
| Support Vector Regressor (SVR) | 0.6715 | $6,758.90 | 28.02% |
| Regresión Lineal (BASE) | 0.5648 | $7,779.61 | 25.40% |

**Modelo Ganador y Justificación:**
El **Random Forest Regressor** fue elegido como el modelo final. Se seleccionó basándose en el **RMSE** y el **$R^2$**, ya que el **RMSE ($4,453.99)$** mide el error en **dólares** (la métrica de negocio más relevante), penalizando fuertemente los errores grandes en los costos más altos.

---

## Conclusión

El análisis de la **Importancia de Variables** del modelo Random Forest reveló los factores más influyentes en el costo del seguro médico:

1.  **Estado de Fumador (44% total)**: Es el factor de riesgo más significativo, duplicando o triplicando los costos estimados.
2.  **Edad (37.1%)**: El factor individual más influyente. El costo se correlaciona fuertemente con la edad del asegurado.
3.  **BMI (10.1%)**: Contribuye significativamente como indicador de riesgo de salud.

El modelo final es robusto, con un **$R^2$ del 85.74%**, lo que proporciona una estimación de gastos médicos de alta precisión.

