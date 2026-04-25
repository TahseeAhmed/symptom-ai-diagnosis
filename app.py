import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
from fpdf import FPDF

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/model.pkl")
features = joblib.load("model/features.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏥 AI Disease Prediction System")

st.write("Select symptoms below:")

# -----------------------------
# SYMPTOM INPUT
# -----------------------------
selected_symptoms = []

for feature in features:
    if st.checkbox(feature.replace("_", " ")):
        selected_symptoms.append(feature)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict"):

    if len(selected_symptoms) == 0:
        st.error("⚠ Please select at least one symptom")
    else:

        input_data = [1 if f in selected_symptoms else 0 for f in features]

        input_df = pd.DataFrame([input_data], columns=features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        top_idx = np.argsort(probabilities)[-5:][::-1]
        diseases = model.classes_

        results = [(diseases[i], round(probabilities[i]*100, 2)) for i in top_idx]

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success(f"🧠 Predicted Disease: {prediction}")

        st.subheader("📊 Top Probabilities")

        df = pd.DataFrame(results, columns=["Disease", "Probability %"])
        st.bar_chart(df.set_index("Disease"))

        # -----------------------------
        # DESCRIPTION
        # -----------------------------
        try:
            import wikipedia
            desc = wikipedia.summary(str(prediction) + " disease", sentences=2)
        except:
            desc = "No detailed description available."

        st.subheader("🧠 Description")
        st.write(desc)

        # -----------------------------
        # DOCTOR
        # -----------------------------
        if "skin" in str(prediction).lower():
            doctor = "Dermatologist"
        elif "heart" in str(prediction).lower():
            doctor = "Cardiologist"
        elif "brain" in str(prediction).lower():
            doctor = "Neurologist"
        else:
            doctor = "General Physician"

        st.subheader("👨‍⚕️ Recommended Doctor")
        st.write(doctor)

        # -----------------------------
        # DOWNLOAD PDF
        # -----------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "AI Disease Report", ln=True)
        pdf.cell(200, 10, f"Disease: {prediction}", ln=True)
        pdf.cell(200, 10, f"Doctor: {doctor}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button("📄 Download Report", f, file_name="report.pdf")
