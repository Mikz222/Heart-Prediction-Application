import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")  # ensure filename matches
model = load_model()

# ==========================
# Streamlit Page Config
# ==========================
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# ==========================
# Custom CSS Clinic Theme
# ==========================
st.markdown("""
    <style>
    /* Force light mode background */
    body {
        background: linear-gradient(135deg, #E8F9F6, #FFFFFF);
        color: #004D40;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Main Title */
    h1 {
        color: #006D77 !important;
        text-align: center;
        font-weight: bold;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #48C6EF, #6F86D6);
        color: white;
        border-radius: 20px;
        height: 3.2em;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6F86D6, #48C6EF);
        color: #004D40;
        transform: scale(1.05);
    }
    /* Result Cards */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }
    .positive {
        background-color: #E6FFFA;
        border: 3px solid #20C997;
        color: #006D77;
    }
    .negative {
        background-color: #FFF0F0;
        border: 3px solid #E63946;
        color: #B22222;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F1FAFE;
        border-right: 2px solid #E0F7FA;
    }
    section[data-testid="stSidebar"] h2 {
        color: #006D77;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Header
# ==========================
st.title("❤️ Heart Disease Prediction App")
st.markdown("### 🩺 Enter patient details below to predict the likelihood of heart disease.")

st.info("ℹ️ **Note:** In the inputs below, `0` = No / False, `1` = Yes / True.\n\n"
        "- **Current Smoker:** 1 if patient currently smokes, 0 if not.\n"
        "- **BPMeds:** 1 if taking blood pressure medication, 0 if not.\n"
        "- **Prevalent Stroke:** 1 if history of stroke, 0 if not.\n"
        "- **Prevalent Hyp:** 1 if hypertension is present, 0 if not.\n"
        "- **Diabetes:** 1 if diabetic, 0 if not.")

# ==========================
# Input Form
# ==========================
def user_input():
    age = st.number_input("Age", min_value=20, max_value=100, value=40)
    gender = st.selectbox("Gender", ["male", "female"])
    education = st.selectbox("Education Level", ["1", "2", "3", "4"])
    currentSmoker = st.selectbox("Current Smoker (1=Yes, 0=No)", [0, 1])
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=60, value=0)
    BPMeds = st.selectbox("On Blood Pressure Medication (1=Yes, 0=No)", [0, 1])
    prevalentStroke = st.selectbox("History of Stroke (1=Yes, 0=No)", [0, 1])
    prevalentHyp = st.selectbox("Hypertension (1=Yes, 0=No)", [0, 1])
    diabetes = st.selectbox("Diabetes (1=Yes, 0=No)", [0, 1])
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level", min_value=40, max_value=300, value=80)

    data = {
        "age": age,
        "Gender": gender,
        "education": education,
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

    return pd.DataFrame([data])

# ==========================
# Prediction
# ==========================
input_df = user_input()

if st.button("🔍 Predict"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.markdown("### 📊 Prediction Result:")
    if pred == 1:
        st.markdown(f'<div class="result-card negative">⚠️ Likely Heart Disease (Confidence: {proba*100:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-card positive">✅ Unlikely Heart Disease (Confidence: {(1-proba)*100:.2f}%)</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Entered Patient Data")
    st.dataframe(input_df, use_container_width=True)
