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
# Custom CSS - Enhanced
# ==========================
st.markdown("""
<style>
/* Global text */
body, p, div, label {
    font-family: 'Segoe UI', sans-serif;
    color: #000 !important;
    font-size: 16px;
    line-height: 1.5;
}

/* Titles */
h1, h2, h3 {
    color: #0D47A1 !important;
    font-weight: 700 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #1E88E5, #1565C0) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 16px 36px !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #42A5F5, #1E88E5) !important;
    transform: scale(1.08);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.35);
}

/* Cards Grid */
.cards-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 25px;
    margin-top: 25px;
}

/* Individual Card */
.card {
    background-color: #ffffff;
    border-radius: 14px;
    padding: 25px;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.12);
    min-height: 400px; /* Equal height */
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 4px 6px 18px rgba(0,0,0,0.25);
}

/* Card Titles */
.card h3 {
    color: #1565C0 !important;
    margin-bottom: 15px;
    font-size: 22px;
}

/* Card content */
.card ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
.card li {
    margin: 12px 0;
    font-size: 16px;
}
.card li strong {
    color: #0D47A1;
}
.card a {
    color: #1565C0;
    text-decoration: none;
    font-weight: 600;
}
.card a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Main Page Content
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

# ==========================
# Info Section with Equal Cards
# ==========================
st.markdown('<div class="cards-container">', unsafe_allow_html=True)

# Heart Health Tips
st.markdown("""
<div class="card">
<h3>💡 Heart Health Tips</h3>
<ul>
<li>🥗 <strong>Eat a balanced diet</strong> → <a href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet" target="_blank">WHO: Healthy Diet</a></li>
<li>🏃 <strong>Exercise regularly</strong> → <a href="https://www.cdc.gov/physical-activity-basics/guidelines/adults.html" target="_blank">CDC Guidelines</a></li>
<li>🚭 <strong>Avoid smoking & alcohol</strong> → <a href="https://www.cdc.gov/tobacco/quit_smoking/index.htm" target="_blank">Quit Smoking - CDC</a></li>
<li>🩺 <strong>Monitor BP, cholesterol & sugar</strong> → <a href="https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502" target="_blank">Mayo Clinic Guide</a></li>
<li>👨‍⚕️ <strong>Regular checkups are essential</strong></li>
</ul>
</div>
""", unsafe_allow_html=True)

# Risk Factors
st.markdown("""
<div class="card">
<h3>⚡ Risk Factors to Watch</h3>
<ul>
<li>🩸 <strong>High blood pressure</strong> → <a href="https://www.heart.org/en/health-topics/high-blood-pressure" target="_blank">AHA Guide</a></li>
<li>🧬 <strong>High cholesterol</strong> → <a href="https://www.cdc.gov/cholesterol/facts.htm" target="_blank">CDC Facts</a></li>
<li>🍩 <strong>Diabetes / pre-diabetes</strong> → <a href="https://diabetes.org/" target="_blank">Diabetes.org</a></li>
<li>🚬 <strong>Smoking & alcohol</strong> → <a href="https://www.cdc.gov/alcohol/fact-sheets/alcohol-use.htm" target="_blank">CDC Alcohol Facts</a></li>
<li>⚖️ <strong>Obesity & inactivity</strong> → <a href="https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight" target="_blank">WHO: Obesity Facts</a></li>
<li>👨‍👩‍👧 <strong>Family history</strong> → <a href="https://www.nhlbi.nih.gov/health/heart-disease" target="_blank">NIH Info</a></li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


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
# Info Section (Side by Side Equal Cards)
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💡 Heart Health Tips")
    st.markdown("""
    <div class="card">
    🥗 Eat a balanced diet → <a href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet" target="_blank">WHO: Healthy Diet</a><br><br>
    🏃 Exercise regularly → <a href="https://www.cdc.gov/physical-activity-basics/guidelines/adults.html" target="_blank">CDC Guidelines</a><br><br>
    🚭 Avoid smoking & alcohol → <a href="https://www.cdc.gov/tobacco/quit_smoking/index.htm" target="_blank">Quit Smoking - CDC</a><br><br>
    🩺 Monitor BP, cholesterol & sugar → <a href="https://www.mayoclinic.org/diseases-conditions/heart-disease/in-depth/heart-disease-prevention/art-20046502" target="_blank">Mayo Clinic Guide</a><br><br>
    👨‍⚕️ Regular checkups are essential.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### ⚡ Risk Factors to Watch")
    st.markdown("""
    <div class="card">
    🩸 High blood pressure → <a href="https://www.heart.org/en/health-topics/high-blood-pressure" target="_blank">AHA Guide</a><br><br>
    🧬 High cholesterol → <a href="https://www.cdc.gov/cholesterol/facts.htm" target="_blank">CDC Facts</a><br><br>
    🍩 Diabetes / pre-diabetes → <a href="https://diabetes.org/" target="_blank">Diabetes.org</a><br><br>
    🚬 Smoking & alcohol → <a href="https://www.cdc.gov/alcohol/fact-sheets/alcohol-use.htm" target="_blank">CDC Alcohol Facts</a><br><br>
    ⚖️ Obesity & inactivity → <a href="https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight" target="_blank">WHO: Obesity Facts</a><br><br>
    👨‍👩‍👧 Family history → <a href="https://www.nhlbi.nih.gov/health/heart-disease" target="_blank">NIH Info</a>
    </div>
    """, unsafe_allow_html=True)

