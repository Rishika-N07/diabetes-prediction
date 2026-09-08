import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("diabetes_model.pkl")

# Page settings
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the patient's health information below "
    "to get a machine learning prediction."
)

st.divider()

# Patient Information
st.subheader("📋 Patient Information")

# Two columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=2
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=79
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.3
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

st.divider()

# Predict button
if st.button("🔮 Predict", use_container_width=True):

    # Create DataFrame
    patient = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # Prediction
    prediction = model.predict(patient)

    # Probability
    probability = model.predict_proba(patient)

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 0:
        st.success("### ✅ No Diabetes Predicted")
    else:
        st.error("### ⚠️ Diabetes Predicted")

    st.write(
        "No Diabetes Probability:",
        f"{probability[0][0] * 100:.2f}%"
    )

    st.write(
        "Diabetes Probability:",
        f"{probability[0][1] * 100:.2f}%"
    )

st.divider()

st.caption(
    "⚠️ This application is for educational purposes only "
    "and is not a medical diagnosis."
)