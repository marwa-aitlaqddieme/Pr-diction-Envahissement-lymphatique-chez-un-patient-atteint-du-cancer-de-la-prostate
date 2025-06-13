import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# Charger le modèle
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🔬 Prédiction : Envahissement lymphatique chez un patient atteint du cancer de la prostate")

# Interface utilisateur
age = st.slider("Âge", 0, 100)
psa = st.number_input("Taux de PSA")
gleason = st.selectbox("Score de Gleason", [6, 7, 8, 9, 10])
tumor_volume = st.number_input("Volume tumoral (en cc)")
capsular_penetration = st.radio("Pénétration capsulaire ?", ["Oui", "Non"])
capsular_penetration = 1 if capsular_penetration == "Oui" else 0

# Données d'entrée
features = np.array([[age, psa, gleason, tumor_volume, capsular_penetration]])

# Prédiction
prob = model.predict_proba(features)[0][1]
st.subheader("🧠 Résultat de la prédiction")
st.write(f"Risque estimé d’envahissement lymphatique : **{prob:.2%}**")

# Rapport téléchargeable
rapport = f"""
Rapport de prédiction :

Âge : {age}
PSA : {psa}
Score de Gleason : {gleason}
Volume tumoral : {tumor_volume}
Pénétration capsulaire : {"Oui" if capsular_penetration else "Non"}

Risque d’envahissement lymphatique : {prob:.2%}
"""
st.download_button("📥 Télécharger le rapport", rapport, file_name="rapport.txt")

# SHAP
st.subheader("📊 Explication du modèle (SHAP)")

@st.cache_resource
def get_explainer():
    return shap.Explainer(model, feature_names=["age", "psa", "gleason", "tumor_volume", "capsular_penetration"])

explainer = get_explainer()
shap_values = explainer(features)

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=5, show=False)
st.pyplot(fig)

# Avertissement médical
st.markdown("---")
st.warning("⚠️ Cette prédiction est une aide à la décision. Elle ne remplace pas un avis médical.")
