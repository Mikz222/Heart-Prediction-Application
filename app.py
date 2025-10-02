import streamlit as st
import pandas as pd
import joblib

# ------------------- LOAD MODEL + SCALER -------------------
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------- STREAMLIT SETTINGS -------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    /* Background */
    .main {
        background: linear-gradient(135deg, #FAF3E0 60%, #F5E0C3);
    }
    /* Titles */
    h1, h2, h3, h4 {
        color: #4B0000;
        font-family: 'Georgia', serif;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #3E0E02;
        color: white;
    }
    section[data-testid="stSidebar"] .stSlider label, 
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #FFD700;
        font-weight: bold;
    }
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #8B0000, #B22222);
        color: white;
        border-radius: 15px;
        height: 3em;
        font-size: 16px;
        border: none;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #B22222, #8B0000);
        color: #FFD700;
    }
    /* Result Cards */
    .result-card {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-top: 20px;
        font-size: 22px;
        font-weight: bold;
        font-family: 'Trebuchet MS', sans-serif;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.25);
    }
    .good {
        background: #FFF9E6;
        color: #155724;
        border: 3px solid #FFD700;
    }
    .bad {
        background: #FCE8E6;
        color: #721c24;
        border: 3px solid #B22222;
    }
    /* Info box */
    .info-box {
        padding: 15px;
        border-left: 6px solid #8B0000;
        background: #FFF8F0;
        margin-top: 15px;
        border-radius: 10px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("<h3 style='color:#8B0000;'>A refined tool for predicting premium wine quality</h3>", unsafe_allow_html=True)

st.markdown("""
Welcome to the **Wine Quality Prediction App**!  
This app predicts whether a wine is **Good Quality (1)** or **Not Good Quality (0)**  
based on its **chemical attributes** using a trained **Random Forest Classifier**.
""")

# ------------------- SIDEBAR -------------------
st.sidebar.title("⚙️ Input Wine Measurements")
st.sidebar.markdown("✨ Use the sliders to adjust the wine’s chemical attributes:")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 1.5, 0.7)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 15.0, 1.9)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.076)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 0.0, 80.0, 11.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.9978)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.slider("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 9.4)

# ------------------- PREDICTION -------------------
if st.sidebar.button("🍇 Predict Quality"):
    features = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    
    # Scale features
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]

    # Result
    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>✅ Excellent! This wine is predicted to be <br><span style='font-size:28px;'>Good Quality 🍷</span><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Unfortunately, this wine is predicted to be <br><span style='font-size:28px;'>Not Good Quality</span><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Explanation Box
    st.markdown("""
    <div class='info-box'>
    🔎 <b>Interpretation:</b>  
    - <b>1 (Good Quality)</b>: Wine meets premium quality standards (balanced acidity, optimal alcohol, etc.)  
    - <b>0 (Not Good Quality)</b>: Wine does not meet premium standards (imbalance in chemical properties).  
    <br>
    Confidence indicates how certain the model is about its prediction.
    </div>
    """, unsafe_allow_html=True)

    # Show entered values
    st.markdown("### 📌 Your Entered Measurements")
    df = pd.DataFrame(features, columns=[
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ])
    st.dataframe(df, use_container_width=True)

# ------------------- EXTRA INFO -------------------
with st.expander("ℹ️ Learn More About Wine Features"):
    st.write("""
    - **Fixed Acidity**: Non-volatile acids that do not evaporate easily.  
    - **Volatile Acidity**: Higher levels can lead to unpleasant vinegar taste.  
    - **Citric Acid**: Adds freshness and flavor.  
    - **Residual Sugar**: Sugar left after fermentation; affects sweetness.  
    - **Chlorides**: Excess can cause salty taste.  
    - **Sulfur Dioxide (Free & Total)**: Preservatives that prevent spoilage.  
    - **Density**: Relation to alcohol & sugar content.  
    - **pH**: Acidity level; balanced wines are usually around 3–4.  
    - **Sulphates**: Contribute to wine preservation and flavor.  
    - **Alcohol**: Higher alcohol can enhance body and perceived quality.  
    """)
