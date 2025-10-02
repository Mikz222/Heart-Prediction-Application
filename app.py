import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")

model = load_model()

# ==========================
# Custom CSS
# ==========================
st.markdown("""
<style>
/* Global */
body, p, div, label {
    font-family: 'Segoe UI', sans-serif;
    color: #000000 !important;
}

/* Title */
h1, h2, h3 {
    color: #0D47A1 !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f9f9f9 !important;
    padding: 20px;
    box-shadow: 2px 0 8px rgba(0,0,0,0.1);
}

/* Input fields */
[data-baseweb="input"] input,
[data-baseweb="select"] select {
    background-color: white !important;
    border: 1px solid #ccc !important;
    border-radius: 8px !important;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1) !important;
    font-size: 15px !important;
    padding: 10px !important;
    transition: all 0.3s ease;
}
[data-baseweb="input"] input:focus,
[data-baseweb="select"] select:focus {
    border-color: #0D47A1 !important;
    box-shadow: 0 0 10px rgba(13,71,161,0.3) !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 14px 30px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
    transform: scale(1.06);
    box-shadow: 4px 4px 14px rgba(0,0,0,0.3);
}

/* Result Box */
.result-box {
    padding: 25px;
    border-radius: 12px;
    background: #0D47A1;
    color: white !important;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin: 20px 0;
    box-shadow: 3px 3px 12px rgba(0,0,0,0.2);
    animation: fadeIn 0.8s ease-in-out;
}

/* Info Cards - Equal Height */
.card-container {
    display: flex;
    gap: 20px;
}
.card {
    flex: 1;
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.15);
    min-height: 360px; /* Force equal size */
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 4px 4px 16px rgba(0,0,0,0.25);
}

/* Smooth Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
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
# Main Page
# ==========================
st.title("💙 Heart Disease Prediction App")
st.markdown("⚠️ *This tool provides insights but does not replace professional medical advice.*")

if st.button("🔍 Predict Heart Disease"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown(f"<div class='result-box'>⚠️ High Risk: Heart disease likely<br>🔢 Risk Probability: {proba*100:.2f}%<br>💡 Recommendation: Please consult a cardiologist immediately.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box'>✅ Low Risk: Heart disease unlikely<br>🔢 Risk Probability: {(1-proba)*100:.2f}%<br>💡 Recommendation: Maintain healthy lifestyle habits.</div>", unsafe_allow_html=True)

    with st.expander("📋 Patient Data Entered"):
        st.dataframe(input_df, use_container_width=True)

# ==========================
# Info Section (Equal Height Cards)
# ==========================
st.markdown("<div class='card-container'>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>💡 Heart Health Tips</h3>
🥗 Eat a balanced diet → <a href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet" target="_blank">WHO: Healthy Diet</a><br><br>
🏃 Exercise regularly → <a href="https://www.cdc.gov/physical-activity-basics/guidelines/adults.html" target="_blank">CDC Guidelines</a><br><br>
🚭 Avoid smoking & alcohol → <a href="https://www.cdc.gov/tobacco/quit_smoking/index.htm" target="_blank">Quit Smoking - CDC</a><br><br>
🩺 Monitor BP, cholesterol & sugar → <a href="https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502" target="_blank">Mayo Clinic Guide</a><br><br>
👨‍⚕️ Regular checkups are essential.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h3>⚡ Risk Factors to Watch</h3>
🩸 High blood pressure → <a href="https://www.heart.org/en/health-topics/high-blood-pressure" target="_blank">AHA Guide</a><br><br>
🧬 High cholesterol → <a href="https://www.cdc.gov/cholesterol/facts.htm" target="_blank">CDC Facts</a><br><br>
🍩 Diabetes / pre-diabetes → <a href="https://diabetes.org/" target="_blank">Diabetes.org</a><br><br>
🚬 Smoking & alcohol → <a href="https://www.cdc.gov/alcohol/fact-sheets/alcohol-use.htm" target="_blank">CDC Alcohol Facts</a><br><br>
⚖️ Obesity & inactivity → <a href="https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight" target="_blank">WHO: Obesity Facts</a><br><br>
👨‍👩‍👧 Family history → <a href="https://www.nhlbi.nih.gov/health/heart-disease" target="_blank">NIH Info</a>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
