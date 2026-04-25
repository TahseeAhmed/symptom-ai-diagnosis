import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os
from fpdf import FPDF

# -----------------------------
# PAGE CONFIG (PROFESSIONAL UI)
# -----------------------------
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f4f7ff;
}
.stButton>button {
    background-color: #4f46e5;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
}
h1 {
    color: #4f46e5;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL (SAFE LOADING)
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    features = joblib.load("model/features.pkl")
    return model, features

model, features = load_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("🏥 AI Disease Prediction System")
st.write("Select symptoms below and get AI-powered prediction")

# -----------------------------
# SYMPTOMS INPUT
# -----------------------------
st.subheader("Select Symptoms")

selected_features = []

cols = st.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        if st.checkbox(feature.replace("_", " ")):
            selected_features.append(1)
        else:
            selected_features.append(0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict Disease"):

    if sum(selected_features) == 0:
        st.error("Please select at least one symptom")
    else:

        input_df = pd.DataFrame([selected_features], columns=features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        top_idx = np.argsort(probabilities)[-5:][::-1]
        diseases = model.classes_

        results = [(diseases[i], round(probabilities[i] * 100, 2)) for i in top_idx]

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success(f"Predicted Disease: {prediction}")

        st.subheader("🧠 Top Predictions")

        df = pd.DataFrame(results, columns=["Disease", "Probability %"])
        st.bar_chart(df.set_index("Disease"))

        # -----------------------------
        # DOCTOR
        # -----------------------------
        def recommend(name):
            name = str(name).lower()
            if "skin" in name:
                return "Dermatologist"
            elif "heart" in name:
                return "Cardiologist"
            elif "brain" in name:
                return "Neurologist"
            else:
                return "General Physician"

        st.info(f"👨‍⚕️ Recommended Doctor: {recommend(prediction)}")

        # -----------------------------
        # DESCRIPTION
        # -----------------------------
        st.subheader("📖 Description")
        st.write(f"{prediction} is detected by AI analysis of symptoms.")

        # -----------------------------
        # PDF DOWNLOAD
        # -----------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "AI Disease Report", ln=True)
        pdf.cell(200, 10, f"Disease: {prediction}", ln=True)

        file_path = "report.pdf"
        pdf.output(file_path)

        with open(file_path, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")
