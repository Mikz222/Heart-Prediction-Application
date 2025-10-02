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
   /* Background */
body {
    background: linear-gradient(135deg, #E8F9F6, #FFFFFF);
    font-family: 'Segoe UI', sans-serif;
}

/* Titles */
h1, h2, h3 {
    color: #006D77;
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
# Title & Instructions
# ==========================
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This is an **AI-powered tool** that helps estimate the likelihood of heart disease  
based on health information you provide.  

🔎 **How to use:**  
1. Fill in the patient’s medical details.  
2. Click **Predict Heart Disease**.  
3. See whether the patient is most likely to **have** or **not have** heart disease.  

⚠️ **Note:** This is a support tool only, not a substitute for professional medical advice.
""")

# ==========================
# Input Form
# ==========================
def user_input():
    st.subheader("🩺 Patient Medical Information")

    age = st.number_input("Age (years)", min_value=20, max_value=100, value=40)

    gender = st.selectbox("Gender", ["Male", "Female"])

    education = st.selectbox("Education Level", ["1 - Primary", "2 - Secondary", "3 - College", "4 - Graduate"])

    currentSmoker = st.selectbox("Currently Smokes?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    cigsPerDay = st.number_input("Cigarettes smoked per day", min_value=0, max_value=60, value=0)

    BPMeds = st.selectbox("Taking Blood Pressure Medication?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentStroke = st.selectbox("History of Stroke?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    prevalentHyp = st.selectbox("Diagnosed with Hypertension?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    diabetes = st.selectbox("Has Diabetes?", [0, 1])
    st.caption("0 = No, 1 = Yes")

    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=150, value=80)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=300, value=80)

    data = {
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
            f"<div class='result-card positive'>⚠️ The model predicts: <br><span style='font-size:26px;'>Most likely have heart disease</span><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card negative'>✅ The model predicts: <br><span style='font-size:26px;'>Most likely don’t have heart disease</span><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    st.markdown("### 📝 Entered Patient Data")
    st.dataframe(input_df, use_container_width=True)

