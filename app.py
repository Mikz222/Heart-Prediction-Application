import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Trained Model
# ==========================
@st.cache_resource
def load_model():
    return joblib.load("heart_nb_pipeline.pkl")  # adjust filename if needed

model = load_model()

# ==========================
# Custom CSS (White + Blue Theme)
# ==========================
st.markdown("""
<style>
/* Global */
body, p, div, label {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px !important;
    color: #0D47A1 !important; /* Deep blue text */
}

/* Title */
h1, h2, h3 {
    color: #0D47A1 !important;  
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0D47A1 !important; 
    color: white !important;
}
[data-testid="stSidebar"] label {
    color: white !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Buttons */
div.stButton > button {
    background-color: #0D47A1 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 18px !important;
    border: none !important;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #1565C0 !important;
    transform: scale(1.03);
}

/* Result Box */
.result-box {
    padding: 20px;
    border-radius: 12px;
    background: #0D47A1;
    color: white !important;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin: 20px 0;
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
# Main Page Layout
# ==========================
st.title("💙 Heart Disease Prediction App")
st.markdown("""
Enter patient details in the sidebar to estimate the risk of heart disease.  
⚠️ *Note: This tool does not replace medical advice.*
""")

# --- Prediction ---
if st.button("🔍 Predict Heart Disease"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.markdown(f"<div class='result-box'>⚠️ High Risk: Heart disease likely<br>Confidence: {proba*100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box'>✅ Low Risk: Heart disease unlikely<br>Confidence: {(1-proba)*100:.2f}%</div>", unsafe_allow_html=True)

    with st.expander("📋 Patient Data Entered"):
        st.dataframe(input_df, use_container_width=True)

# ==========================
# Info Sections with Links
# ==========================
st.markdown("### 💡 Heart Health Tips")
st.write("""
- 🥗 **Eat a balanced diet** rich in fruits, vegetables, and whole grains.  
  👉 [Learn more (WHO)](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)  

- 🏃 **Exercise regularly** (at least 30 minutes per day).  
  👉 [See CDC Guidelines](https://www.cdc.gov/physical-activity-basics/guidelines/adults.html)  

- 🚭 **Avoid smoking and limit alcohol intake**.  
  👉 [Quit Smoking Help (CDC)](https://www.cdc.gov/tobacco/quit_smoking/index.htm)  

- 🩺 **Monitor blood pressure, cholesterol, and blood sugar regularly**.  
  👉 [Mayo Clinic: Heart Disease Prevention](https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502)  

- 👨‍⚕️ **Visit your doctor for regular checkups**.  
""")

st.markdown("### ⚡ Risk Factors to Watch")
st.write("""
- 🩸 **High blood pressure (Hypertension)**  
  👉 [American Heart Association: Hypertension](https://www.heart.org/en/health-topics/high-blood-pressure)  

- 🧬 **High cholesterol (Hyperlipidemia)**  
  👉 [CDC: Cholesterol Facts](https://www.cdc.gov/cholesterol/facts.htm)  

- 🍩 **Diabetes or pre-diabetes**  
  👉 [Diabetes.org](https://diabetes.org/)  

- 🚬 **Smoking and excessive alcohol intake**  
  👉 [CDC: Alcohol and Your Health](https://www.cdc.gov/alcohol/fact-sheets/alcohol-use.htm)  

- ⚖️ **Obesity and sedentary lifestyle**  
  👉 [WHO: Obesity and Overweight](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight)  

- 👨‍👩‍👧‍👦 **Family history of heart disease**  
  👉 [NIH: Family History and Heart Disease](https://www.nhlbi.nih.gov/health/heart-disease)  
""")

st.markdown("### 📞 Contact Your Doctor")
st.info("""
If you experience symptoms like **chest pain, shortness of breath, dizziness, or irregular heartbeat**,  
please **consult a healthcare professional immediately**.
""")
