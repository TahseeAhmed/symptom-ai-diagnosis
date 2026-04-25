import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7ff;
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
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='title'>🏥 AI Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart AI-powered symptom checker with medical insights</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
try:
    model = joblib.load("model/model.pkl")
    features = joblib.load("model/features.pkl")
except:
    st.error("❌ Model files missing or corrupted. Please re-upload model.pkl and features.pkl")
    st.stop()

# -----------------------------
# LAYOUT
# -----------------------------
col1, col2 = st.columns([2, 1])

# -----------------------------
# SYMPTOMS SECTION
# -----------------------------
with col1:
    st.markdown("## 🧠 Select Your Symptoms")

    selected_symptoms = []

    for feature in features:
        if st.checkbox(feature.replace("_", " ")):
            selected_symptoms.append(feature)

    predict = st.button("🔍 Predict Disease")

# -----------------------------
# INFO PANEL
# -----------------------------
with col2:
    st.markdown("## 📊 Info Panel")
    st.info("Select symptoms on the left and click Predict to get AI diagnosis")

# -----------------------------
# PREDICTION
# -----------------------------
if predict:

    if len(selected_symptoms) == 0:
        st.warning("⚠ Please select at least one symptom")

    else:

        input_data = [1 if f in selected_symptoms else 0 for f in features]
        input_df = pd.DataFrame([input_data], columns=features)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        top_idx = np.argsort(probabilities)[-5:][::-1]
        diseases = model.classes_

        results = pd.DataFrame({
            "Disease": [diseases[i] for i in top_idx],
            "Probability": [probabilities[i]*100 for i in top_idx]
        })

        # -----------------------------
        # RESULT HEADER
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
            text_auto=".2f",
            color="Probability",
            title="Top 5 Disease Probabilities"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # DESCRIPTION
        # -----------------------------
        st.markdown("## 🧠 Medical Description")

        try:
            import wikipedia
            desc = wikipedia.summary(str(prediction) + " disease", sentences=2)
        except:
            desc = "No detailed medical description available for this prediction."

        st.write(desc)

        # -----------------------------
        # DOCTOR RECOMMENDATION
        # -----------------------------
        if "heart" in str(prediction).lower():
            doctor = "❤️ Cardiologist"
        elif "skin" in str(prediction).lower():
            doctor = "🧴 Dermatologist"
        elif "brain" in str(prediction).lower():
            doctor = "🧠 Neurologist"
        else:
            doctor = "👨‍⚕️ General Physician"

        st.markdown("## 🏥 Recommended Doctor")
        st.success(doctor)

        # -----------------------------
        # DOWNLOAD REPORT
        # -----------------------------
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, "AI Disease Prediction Report", ln=True)
        pdf.cell(200, 10, f"Disease: {prediction}", ln=True)
        pdf.cell(200, 10, f"Doctor: {doctor}", ln=True)

        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button(
                "📄 Download Medical Report",
                f,
                file_name="AI_Disease_Report.pdf"
            )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit | AI Healthcare System | Machine Learning Powered")
