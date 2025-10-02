import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")  # make sure filename matches

model = load_model()

# ==========================
# Streamlit Page Settings
# ==========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================
# Sidebar - Patient Input
# ==========================
with st.sidebar:
    st.header("📝 Patient Details")

    age = st.number_input("Age (years)", min_value=20, max_value=100, value=40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["1 - Primary", "2 - Secondary", "3 - College", "4 - Graduate"])

    currentSmoker = st.selectbox("Currently Smokes?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=60, value=0)

    BPMeds = st.selectbox("On Blood Pressure Medication?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentStroke = st.selectbox("History of Stroke?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentHyp = st.selectbox("Hypertension?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    diabetes = st.selectbox("Diabetes?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=300, value=80)

    input_data = {
        "age": age,
        "Gender": gender.lower(),
        "education": education.split(" - ")[0],
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose,
    }

    input_df = pd.DataFrame([input_data])

# ==========================
# Main Page - App Info
# ==========================
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the **Heart Disease Prediction Tool**.  
Enter patient details in the **sidebar** to estimate the likelihood of heart disease.  

⚠️ **Note:** This is a support tool only, not a substitute for professional medical advice.
""")

# ==========================
# Prediction
# ==========================
if st.button("🔍 Predict Heart Disease"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.subheader("📌 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ The model predicts: **Most likely have heart disease**\n\nConfidence: {proba*100:.2f}%")
    else:
        st.success(f"✅ The model predicts: **Most likely don’t have heart disease**\n\nConfidence: {(1-proba)*100:.2f}%")

    st.markdown("### 📋 Patient Data Entered")
    st.dataframe(input_df, use_container_width=True)
