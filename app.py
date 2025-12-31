import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Load trained model & scaler
# ===============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Preventive Care AI", layout="centered")

st.title("🏥 Preventive Care Risk Screening")
st.write("AI-powered early diabetes risk assessment (preventive use only).")

st.divider()

# ===============================
# INPUT FORM
# ===============================
st.subheader("Patient Information")

age = st.slider("Age", 5, 90, 40)
bmi = st.number_input("BMI", 15.0, 50.0, 27.0)
hbA1c = st.number_input("HbA1c Level", 4.0, 10.0, 5.8)
glucose = st.number_input("Blood Glucose Level", 60, 300, 140)

hypertension = st.selectbox("Hypertension History", [0, 1])
heart_disease = st.selectbox("Heart Disease History", [0, 1])

gender = st.selectbox("Gender", ["Female", "Male"])

smoking = st.selectbox(
    "Smoking History",
    ["Never", "Former / Ever", "Current"]
)

smoking_map = {
    "Never": 0,
    "Former / Ever": 1,
    "Current": 2
}
smoking_encoded = smoking_map[smoking]

race = st.selectbox(
    "Race",
    ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
)

# ===============================
# AUTO CLINICAL NOTE GENERATOR
# ===============================
def generate_clinical_note(row, risk):
    notes = []

    if row['bmi'] >= 25:
        notes.append("Overweight BMI")

    if row['hbA1c_level'] >= 5.7:
        notes.append("Elevated HbA1c")

    if row['blood_glucose_level'] >= 140:
        notes.append("Elevated blood glucose")

    if row['hypertension'] == 1:
        notes.append("History of hypertension")

    if row['heart_disease'] == 1:
        notes.append("History of heart disease")

    if risk >= 0.65:
        plan = "High risk for diabetes. Immediate screening and lifestyle intervention advised."
    elif risk >= 0.35:
        plan = "Moderate risk. Preventive screening and lifestyle counseling recommended."
    else:
        plan = "Low immediate risk. Continue routine monitoring."

    note_text = "; ".join(notes) if notes else "No major clinical risk indicators detected"

    return f"{note_text}. {plan}"

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Assess Risk"):

    # EXACT training feature list
    feature_names = [
        'year',
        'gender',
        'age',
        'race:AfricanAmerican',
        'race:Asian',
        'race:Caucasian',
        'race:Hispanic',
        'race:Other',
        'hypertension',
        'heart_disease',
        'smoking_history',
        'bmi',
        'hbA1c_level',
        'blood_glucose_level'
    ]

    # Empty row
    X_input = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Fill numeric values
    X_input['year'] = 2024
    X_input['age'] = age
    X_input['bmi'] = bmi
    X_input['hbA1c_level'] = hbA1c
    X_input['blood_glucose_level'] = glucose
    X_input['hypertension'] = hypertension
    X_input['heart_disease'] = heart_disease
    X_input['smoking_history'] = smoking_encoded

    # Gender encoding (same as training)
    X_input['gender'] = 1 if gender == "Male" else 0

    # Race one-hot (same as training)
    X_input[f"race:{race}"] = 1

    # Scale
    X_scaled = scaler.transform(X_input)

    # ML prediction
    risk_prob = model.predict_proba(X_scaled)[0][1]

    # Risk bucket
    if risk_prob >= 0.65:
        category = "🔴 High Risk"
    elif risk_prob >= 0.35:
        category = "🟠 Moderate Risk"
    else:
        category = "🟢 Low Risk"

    # Auto clinical note
    clinical_summary = generate_clinical_note({
        "bmi": bmi,
        "hbA1c_level": hbA1c,
        "blood_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease
    }, risk_prob)

    # ===============================
    # OUTPUT
    # ===============================
    st.divider()
    st.subheader("🧠 Risk Assessment Result")

    st.metric("Diabetes Risk Probability", f"{risk_prob:.2f}")
    st.write("**Risk Category:**", category)

    st.subheader("📝 AI-Generated Clinical Summary")
    st.info(clinical_summary)

    st.caption(
        "⚠️ Preventive risk estimation only. This tool does not provide a medical diagnosis."
    )
