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
# Custom CSS for Styling
# ==========================
st.markdown("""
<style>
/* Global */
body, p, div, label {
    font-family: 'Roboto', 'Segoe UI', sans-serif;
    font-size: 18px !important;
    color: #2C2C2C !important;
}

/* Title */
h1 {
    font-size: 46px !important;
    font-weight: 800 !important;
    color: #1E3D59 !important;
    text-align: center !important;
    margin-bottom: 20px;
}

/* Subheaders */
h2, h3 {
    color: #E07A5F !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    padding: 20px !important;
    border-right: 3px solid #1E3D59 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #1E3D59 !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #E07A5F !important;
    transform: scale(1.05);
}

/* Cards */
.result-box {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1E3D59, #E07A5F);
    color: white !important;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin: 20px 0;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
.info-card {
    padding: 18px;
    margin: 12px 0;
    border-radius: 12px;
    background-color: #FFFFFF;
    border: 2px solid #E07A5F;
    color: #2C2C2C !important;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

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
    prevalentStroke = st.selectbox("History of Stroke?", [0, 1])
    prevalentHyp = st.selectbox("Hypertension?", [0, 1])
    diabetes = st.selectbox("Diabetes?", [0, 1])

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
# Main Page Layout
# ==========================
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the **Heart Disease Prediction Tool**.  
Enter patient details in the **sidebar** and get instant predictions.  

⚠️ *Note: This is a support tool only, not a substitute for medical advice.*
""")

# --- Prediction ---
if st.button("🔍 Predict Heart Disease"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown(f"<div class='result-box'>⚠️ High Risk: Patient may have heart disease<br>Confidence: {proba*100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box'>✅ Low Risk: Patient unlikely to have heart disease<br>Confidence: {(1-proba)*100:.2f}%</div>", unsafe_allow_html=True)

    # Show Patient Data
    with st.expander("📋 Patient Data Entered"):
        st.dataframe(input_df, use_container_width=True)

# --- Info Sections ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🩺 Heart Health Tips")
    st.markdown("""
    <div class='info-card'>
    ✅ Eat more fruits, vegetables, whole grains. <br>
    ✅ Exercise at least 30 mins daily. <br>
    ✅ Avoid smoking & limit alcohol. <br>
    ✅ Regularly monitor BP, sugar, cholesterol. <br>
    ✅ Visit your doctor for checkups.  
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Risk Factors to Watch")
    st.markdown("""
    <div class='info-card'>
    ⚡ High blood pressure (Hypertension) <br>
    ⚡ High cholesterol <br>
    ⚡ Diabetes / pre-diabetes <br>
    ⚡ Smoking, alcohol <br>
    ⚡ Obesity, sedentary lifestyle <br>
    ⚡ Family history of heart disease  
    </div>
    """, unsafe_allow_html=True)

# --- Contact Section ---
st.markdown("### 📞 Contact Your Doctor")
st.markdown("""
<div class='info-card'>
If you experience symptoms like **chest pain, shortness of breath, dizziness, or irregular heartbeat**,  
please **consult a healthcare professional immediately**.  
</div>
""", unsafe_allow_html=True)
