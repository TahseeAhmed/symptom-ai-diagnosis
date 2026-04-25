import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from fpdf import FPDF

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/model.pkl")
features = joblib.load("model/features.pkl")

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f4f7ff;
        }
        .title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: #4f46e5;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='title'>🏥 AI Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart AI-powered health assistant for symptom analysis</div>", unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([2, 1])

# -----------------------------
# SYMPTOMS SECTION
# -----------------------------
with col1:
    st.markdown("### 🧠 Select Your Symptoms")

    selected_symptoms = []

    with st.container():
        for feature in features:
            if st.checkbox(feature.replace("_", " ")):
                selected_symptoms.append(feature)

    predict_btn = st.button("🔍 Predict Disease")

# -----------------------------
# RESULTS PANEL
# -----------------------------
with col2:
    st.markdown("### 📊 Info Panel")
    st.info("Select symptoms and click Predict to see AI diagnosis")

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if predict_btn:

    if len(selected_symptoms) == 0:
        st.error("⚠ Please select at least one symptom")

    else:

        input_data = [1 if f in selected_symptoms else 0 for f in features]
        input_df = pd.DataFrame([input_data], columns=features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        diseases = model.classes_
        top_idx = np.argsort(probabilities)[-5:][::-1]

        results = pd.DataFrame({
            "Disease": [diseases[i] for i in top_idx],
            "Probability": [probabilities[i]*100 for i in top_idx]
        })

        # -----------------------------
        # MAIN RESULT CARD
        # -----------------------------
        st.markdown("---")
        st.markdown("## 🧬 Diagnosis Result")

        st.success(f"🧠 Predicted Disease: {prediction}")

        # -----------------------------
        # CHART (PROFESSIONAL)
        # -----------------------------
        fig = px.bar(
            results,
            x="Disease",
            y="Probability",
            color="Probability",
            text_auto=".2f",
            title="Top 5 Disease Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # DESCRIPTION
        # -----------------------------
        try:
            import wikipedia
            desc = wikipedia.summary(str(prediction) + " disease", sentences=2)
        except:
            desc = "No detailed medical description available."

        st.markdown("### 🧠 Medical Description")
        st.write(desc)

        # -----------------------------
        # DOCTOR RECOMMENDATION
        # -----------------------------
        if "skin" in str(prediction).lower():
            doctor = "👨‍⚕️ Dermatologist"
        elif "heart" in str(prediction).lower():
            doctor = "❤️ Cardiologist"
        elif "brain" in str(prediction).lower():
            doctor = "🧠 Neurologist"
        else:
            doctor = "👨‍⚕️ General Physician"

        st.markdown("### 🏥 Recommended Doctor")
        st.success(doctor)

        # -----------------------------
        # PDF REPORT
        # -----------------------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "AI Disease Prediction Report", ln=True)
        pdf.cell(200, 10, f"Disease: {prediction}", ln=True)
        pdf.cell(200, 10, f"Doctor: {doctor}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                "📄 Download Full Report",
                f,
                file_name="AI_Disease_Report.pdf"
            )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit + Machine Learning | Designed for AI Healthcare System")
