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
