import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURACIÓN E INICIALIZACIÓN ---

# Cargar el modelo guardado (Random Forest)
# Nota: Este archivo fue creado en el paso 'guardar_modelo.py'
modelo_rf = joblib.load('modelo_random_forest.pkl')

# Cargar el dataset original solo para obtener la estructura de las columnas después de la codificación
# Se asume que 'insurance.csv' está en la misma carpeta
df_base = pd.read_csv('insurance.csv')

# Función para preprocesar los datos de entrada del usuario y codificarlos
# Debe replicar EXACTAMENTE la codificación hecha en la fase de EDA
def preprocess_input(input_data):
    # Crear un DataFrame a partir de los datos de entrada
    df_input = pd.DataFrame([input_data])
    
    # 1. Aplicar One-Hot Encoding a las variables categóricas
    df_encoded = pd.get_dummies(df_input, columns=['sex', 'smoker', 'region'])
    
    # Lista de TODAS las columnas que el modelo fue entrenado a esperar (orden y nombre exacto)
    dummy_cols = [
        'age', 'bmi', 'children', 'sex_female', 'sex_male', 
        'smoker_no', 'smoker_yes', 'region_northeast', 
        'region_northwest', 'region_southeast', 'region_southwest'
    ]

    # Rellenar las columnas faltantes (si el usuario no eligió esa opción, su valor es 0)
    for col in dummy_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reordenar las columnas al orden que espera el modelo
    df_final = df_encoded[dummy_cols]
    
    return df_final

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Predicción de Seguros", layout="centered")
st.title("💰 Predictor de Costos de Seguros Médicos")
st.subheader("Modelo Random Forest Regressor (R²: 85.74%)")

# --- ENTRADAS DEL USUARIO (Barra Lateral) ---
st.sidebar.header("Ingresar Datos del Cliente")

age = st.sidebar.slider("1. Edad", 18, 65, 30)
bmi = st.sidebar.slider("2. Índice de Masa Corporal (BMI)", 15.0, 53.0, 25.0)
children = st.sidebar.selectbox("3. Número de Hijos", [0, 1, 2, 3, 4, 5])
sex = st.sidebar.selectbox("4. Género", ["male", "female"])
smoker = st.sidebar.selectbox("5. Fumador (Factor Clave)", ["yes", "no"])
region = st.sidebar.selectbox("6. Región de Residencia", ["northeast", "northwest", "southeast", "southwest"])

# --- BOTÓN DE PREDICCIÓN ---
if st.button("Calcular Costo Estimado"):
    
    # 1. Crear el diccionario de datos de entrada
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex,
        'smoker': smoker,
        'region': region
    }
    
    # 2. Preprocesar y codificar
    final_input_df = preprocess_input(input_data)
    
    # 3. Realizar la predicción (en escala logarítmica)
    prediction_log = modelo_rf.predict(final_input_df)
    
    # 4. Deshacer la transformación Logarítmica para obtener el costo final en dólares
    final_prediction = np.exp(prediction_log)[0]
    
    st.success("La estimación del costo médico es:")
    st.markdown(f"# ${final_prediction:,.2f}")
    
    st.markdown("---")
    if smoker == "yes":
        st.warning("⚠️ **ALTO RIESGO:** El estatus de fumador es el factor más significativo en el aumento de costos.")
    else:
        st.info("✅ **BAJO RIESGO:** La estimación se beneficia de no ser fumador.")
        