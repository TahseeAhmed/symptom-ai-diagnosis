from flask import Flask, render_template, request, redirect, session, send_file
import joblib
import numpy as np
import pandas as pd
from fpdf import FPDF
import wikipedia

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = joblib.load('model/model.pkl')
features = joblib.load('model/features.pkl')

# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html', features=features)

# -----------------------------
# WIKIPEDIA DESCRIPTION FIXED
# -----------------------------
def get_medical_description(name):
    try:
        query = name + " disease"
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)

        if any(x in summary.lower() for x in ["law", "court", "music"]):
            raise Exception("Wrong context")

        return summary

    except:
        try:
            results = wikipedia.search(name + " disease")
            if results:
                return wikipedia.summary(results[0], sentences=3)
        except:
            pass

    return f"{name} is a medical condition detected by AI."

# -----------------------------
# DOCTOR RECOMMENDATION
# -----------------------------
def recommend_doctor(name):
    name = name.lower()

    if "skin" in name or "fungal" in name:
        return "Dermatologist"
    elif "fever" in name or "flu" in name:
        return "General Physician"
    elif "heart" in name:
        return "Cardiologist"
    elif "brain" in name or "migraine" in name:
        return "Neurologist"
    else:
        return "General Physician"

# -----------------------------
# PREDICT ROUTE (FIXED EMPTY INPUT)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    input_data = []
    selected = False

    for feature in features:
        value = request.form.get(feature)

        if value == 'on':
            input_data.append(1)
            selected = True
        else:
            input_data.append(0)

    # 🚨 BLOCK EMPTY INPUT
    if not selected:
        return "<h2 style='color:red;text-align:center;'>⚠ Please select at least one symptom!</h2>"

    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    top_idx = np.argsort(probabilities)[-5:][::-1]
    diseases = model.classes_

    results = [(diseases[i], round(probabilities[i]*100, 2)) for i in top_idx]

    session['prediction'] = prediction
    session['results'] = results

    return redirect('/result')

# -----------------------------
# RESULT PAGE
# -----------------------------
@app.route('/result')
def result():

    prediction = session.get('prediction')
    results = session.get('results')

    description = get_medical_description(prediction)
    doctor = recommend_doctor(prediction)

    return render_template(
        "result.html",
        prediction=prediction,
        results=results,
        description=description,
        doctor=doctor
    )

# -----------------------------
# PDF DOWNLOAD
# -----------------------------
@app.route('/download')
def download():

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "AI Disease Prediction Report", ln=True)
    pdf.cell(200, 10, f"Disease: {session.get('prediction')}", ln=True)
    pdf.cell(200, 10, f"Doctor: {recommend_doctor(session.get('prediction'))}", ln=True)

    pdf.output("report.pdf")

    return send_file("report.pdf", as_attachment=True)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)