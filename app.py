import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")  # match filename

model = load_model()

# ==========================
# Streamlit Page Settings
# ==========================
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #F8F9FA;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #B22222;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #DC143C, #B22222);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #B22222, #DC143C);
        color: #FFD700;
    }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }
    .positive {
        background-color: #FDECEC;
        border: 3px solid #DC143C;
        color: #8B0000;
    }
    .negative {
        background-color: #E6F9EC;
        border: 3px solid #2E8B57;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Title & Instructions
# ==========================
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the **Heart Disease Prediction Tool**.  
Please enter the patient’s medical details below.  

- The model is trained on health data (age, cholesterol, blood pressure, etc.).  
- Predictions are **probabilities** of having heart disease.  
- **Result**:  
  - ✅ *Unlikely Heart Disease* → Model predicts no risk.  
  - ⚠️ *Likely Heart Disease* → Model predicts possible risk.  

Use this as a **support tool only**. It does not replace professional medical advice.
""")

# ==========================
# Input Form
# ==========================
def user_input():
    st.subheader("🩺 Patient Medical Information")

    age = st.number_input("Age", min_value=20, max_value=100, value=40)

    gender = st.selectbox("Gender", ["male", "female"])

    education = st.selectbox("Education Level (1–4)", ["1", "2", "3", "4"])
    st.caption("1 = Primary, 2 = Secondary, 3 = Tertiary, 4 = Graduate")

    currentSmoker = st.selectbox("Current Smoker", [0, 1])
    st.caption("0 = No, 1 = Yes")

    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=60, value=0)

    BPMeds = st.selectbox("On Blood Pressure Medication", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentStroke = st.selectbox("History of Stroke", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentHyp = st.selectbox("Hypertension", [0, 1])
    st.caption("0 = No, 1 = Yes")

    diabetes = st.selectbox("Diabetes", [0, 1])
    st.caption("0 = No, 1 = Yes")

    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
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

if st.button("🔍 Predict Heart Disease"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.subheader("📌 Prediction Result")
    if pred == 1:
        st.markdown(
            f"<div class='result-card positive'>⚠️ The model predicts: <br><span style='font-size:26px;'>Likely Heart Disease</span><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card negative'>✅ The model predicts: <br><span style='font-size:26px;'>Unlikely Heart Disease</span><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    st.markdown("### 📝 Entered Patient Data")
    st.dataframe(input_df, use_container_width=True)
