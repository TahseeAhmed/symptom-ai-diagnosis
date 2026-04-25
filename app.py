from flask import Flask, render_template, request, redirect, session, send_file
import joblib
import numpy as np
import pandas as pd
import os
import json
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "secret123"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/model.pkl")
features = joblib.load("model/features.pkl")

# -----------------------------
# HOME
# -----------------------------
@app.route('/')
def home():
    return render_template("index.html", features=features)

# -----------------------------
# DESCRIPTION
# -----------------------------
def get_medical_description(name):
    try:
        import wikipedia
        return wikipedia.summary(str(name) + " disease", sentences=2)
    except:
        return f"{name} is detected by AI system."

# -----------------------------
# DOCTOR RECOMMENDATION
# -----------------------------
def recommend_doctor(name):
    name = str(name).lower()

    if "skin" in name:
        return "Dermatologist"
    elif "heart" in name:
        return "Cardiologist"
    elif "brain" in name:
        return "Neurologist"
    else:
        return "General Physician"

# -----------------------------
# PREDICT
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    input_data = []
    selected = False

    for feature in features:
        value = request.form.get(feature)

        if value == "on":
            input_data.append(1)
            selected = True
        else:
            input_data.append(0)

    if not selected:
        return "<h2 style='color:red;text-align:center;'>Select at least one symptom</h2>"

    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    top_idx = np.argsort(probabilities)[-5:][::-1]
    diseases = model.classes_

    results = [(diseases[i], round(probabilities[i]*100, 2)) for i in top_idx]

    session['prediction'] = str(prediction)
    session['results'] = results

    return redirect("/result")

# -----------------------------
# RESULT
# -----------------------------
@app.route('/result')
def result():

    prediction = session.get('prediction', "Unknown")
    results = session.get('results', [])

    description = get_medical_description(prediction)
    doctor = recommend_doctor(prediction)

    return render_template(
        "result.html",
        prediction=prediction,
        results=results,
        description=description,
        doctor=doctor,
        chart_data=json.dumps(results)
    )

# -----------------------------
# PDF DOWNLOAD
# -----------------------------
@app.route('/download')
def download():

    prediction = session.get('prediction', "Unknown")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "AI Disease Prediction Report", ln=True)
    pdf.cell(200, 10, f"Disease: {prediction}", ln=True)
    pdf.cell(200, 10, f"Doctor: {recommend_doctor(prediction)}", ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)

# -----------------------------
# RUN (IMPORTANT FIX)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
