# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.sidebar.image("ehtp_logo.png", width=150)
st.sidebar.title("Mini-projet ML")
st.sidebar.write("Détection des attaques IoT")
st.sidebar.write("Nom : BELMOKHTARE Fatima Ezzahra")
st.sidebar.write("Filière : GIS - EHTP")
# --------------------------
# Chargement du modèle et du scaler
# --------------------------
@st.cache_resource
def load_model():
    model = joblib.load("modele_final.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --------------------------
# Liste des 20 features sélectionnées (SelectKBest)
# --------------------------
selected_features = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_data_pkts_tot',
    'bwd_data_pkts_tot', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'flow_pkts_per_sec',
    'fwd_header_size_tot', 'bwd_header_size_tot', 'flow_FIN_flag_count',
    'flow_SYN_flag_count', 'flow_RST_flag_count', 'fwd_PSH_flag_count',
    'bwd_PSH_flag_count', 'flow_ACK_flag_count', 'bwd_URG_flag_count',
    'active_std', 'idle_min', 'idle_max'
]

# --------------------------
# Titre de l'application
# --------------------------
st.title("Détection d'attaques IoT - Machine Learning")
st.markdown("""
Cette application permet de prédire le type d'attaque dans un réseau IoT.
Vous pouvez :
- **Uploader un fichier CSV** avec toutes les colonnes
- **Entrer manuellement les valeurs** des 20 features les plus importantes
""")

# --------------------------
# Choix du mode
# --------------------------
mode = st.radio("Sélectionnez le mode d'entrée :", ("Upload CSV", "Saisie manuelle"))

# --------------------------
# Mode 1 : Upload CSV
# --------------------------
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :")
        st.dataframe(data.head())

        # Préprocessing automatique : encoder + scaler
        data_encoded = pd.get_dummies(data, drop_first=True)
        data_scaled = scaler.transform(data_encoded)

        predictions = model.predict(data_scaled)
        data['Prediction'] = predictions

        st.write("Résultats des prédictions :")
        st.dataframe(data[['Prediction']].head())

# --------------------------
# Mode 2 : Saisie manuelle
# --------------------------
else:
    st.subheader("Entrez les valeurs des 20 features sélectionnées")
    input_data = {}
    for feat in selected_features:
        val = st.number_input(f"{feat} :", value=0.0, format="%.6f")
        input_data[feat] = val

    if st.button("Prédire"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Type d'attaque prédit : **{prediction[0]}**")