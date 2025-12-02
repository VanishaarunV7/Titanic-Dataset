import streamlit as st
import pandas as pd
import joblib
import time

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)
# ===========================
# FULL LIGHT THEME + ANIMATION
# ===========================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f6f9fc, #e9f3ff, #ffffff);
    background-size: 300% 300%;
    animation: moveBg 12s ease infinite;
    color: #1e293b;
    font-family: 'Segoe UI', sans-serif;
}

/* Animated light gradient */
@keyframes moveBg {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

/* Light floating particle bubbles */
.particle {
    position: fixed;
    border-radius: 50%;
    background: rgba(100, 149, 237, 0.15);
    animation: floatUp linear infinite;
}

@keyframes floatUp {
    0% { transform: translateY(0px); opacity: 0.7; }
    100% { transform: translateY(-900px); opacity: 0; }
}

#p1 { width: 12px; height: 12px; left: 10%; animation-duration: 25s; }
#p2 { width: 8px; height: 8px; left: 25%; animation-duration: 20s; }
#p3 { width: 15px; height: 15px; left: 40%; animation-duration: 22s; }
#p4 { width: 10px; height: 10px; left: 55%; animation-duration: 30s; }
#p5 { width: 20px; height: 20px; left: 70%; animation-duration: 32s; }
#p6 { width: 7px; height: 7px; left: 85%; animation-duration: 28s; }

/* Input boxes light mode */
input, select, .stSelectbox, .stTextInput {
    background-color: white !important;
    color: #1e293b !important;
}

</style>

<div id="p1" class="particle"></div>
<div id="p2" class="particle"></div>
<div id="p3" class="particle"></div>
<div id="p4" class="particle"></div>
<div id="p5" class="particle"></div>
<div id="p6" class="particle"></div>

""", unsafe_allow_html=True)


# ===================================
# CUSTOM CSS — GRADIENT BG + CARDS + ANIMATION
# ===================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
}

.big-title {
    font-size: 42px;
    text-align: center;
    color: #4db8ff;
    font-weight: bold;
    text-shadow: 0 0 20px #60a5fa;
}

.sub-title {
    font-size: 18px;
    text-align: center;
    color: #bde0fe;
    margin-bottom: 25px;
}

.card {
    padding: 20px;
    background-color: #1e293b;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(0,0,0,0.4);
    border: 1px solid #334155;
    margin-bottom: 25px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.01);
    box-shadow: 0 0 35px rgba(96,165,250,0.4);
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 16px;
    color: #93c5fd;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 10px #60a5fa; }
    to { text-shadow: 0 0 20px #bae6fd; }
}

.floating-icon {
    position: fixed;
    bottom: 30px;
    right: 30px;
    font-size: 45px;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}

</style>
""", unsafe_allow_html=True)
<style>
.sidebar .sidebar-content {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(180,180,180,0.4);
    color:#1e293b;
}

.sidebar-title {
    color:#2563eb;
    text-shadow: 0 0 5px #93c5fd;
}

.section-title {
    color:#3b82f6;
}

.glass-card {
    background: rgba(255,255,255,0.7);
    border:1px solid rgba(200,200,200,0.6);
    border-radius:12px;
    padding:12px;
}

.icon-btn {
    background: rgba(255,255,255,0.8);
    color:#1e293b;
    border:1px solid rgba(200,200,200,0.7);
}
.icon-btn:hover{
    background:#e0f2fe;
}
</style>


# ==========================
# SUPER PREMIUM STYLISH SIDEBAR
# ==========================

# Sidebar custom CSS
st.markdown("""
<style>

.sidebar .sidebar-content {
    background: linear-gradient(180deg, #0f172a, #1e293b, #0f172a);
    animation: bgMove 10s ease infinite;
    background-size: 200% 200%;
    padding: 20px;
}

@keyframes bgMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Stylish headings */
.sidebar-title {
    font-size: 26px;
    color: #38bdf8;
    font-weight: bold;
    text-shadow: 0 0 15px #60a5fa;
    margin-bottom: 10px;
}

/* Section headers */
.section-title {
    font-size: 20px;
    margin-top: 18px;
    color: #93c5fd;
    font-weight: bold;
    text-shadow: 0 0 8px #38bdf8;
}

/* Glass card blocks */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
    backdrop-filter: blur(10px);
}

/* Stylish icon buttons */
.icon-btn {
    display: inline-block;
    padding: 10px 15px;
    margin-right: 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    text-decoration: none;
    color: #e0f2fe;
    transition: 0.3s;
    font-size: 15px;
}
.icon-btn:hover {
    background: rgba(96,165,250,0.3);
    transform: scale(1.08);
    box-shadow: 0 0 8px #38bdf8;
}

.team-name {
    font-size: 18px;
    font-weight: bold;
    color: #7dd3fc;
}

.team-role {
    color: #cbd5e1;
    font-size: 14px;
    margin-bottom: 6px;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTENT
with st.sidebar:

    st.markdown("<p class='sidebar-title'>🚢 Titanic ML Project</p>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>📘 About</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    This is an advanced Machine Learning project predicting Titanic passenger survival 
    using a Random Forest model — with a beautifully animated UI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🛠 Tools Used</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    • Python  
    • Pandas  
    • Scikit-Learn  
    • Random Forest  
    • Streamlit  
    • GitHub  
    • Cloud Deployment  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🔬 ML Workflow</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    1️⃣ Data Cleaning  
    2️⃣ Feature Engineering  
    3️⃣ Encoding  
    4️⃣ Scaling  
    5️⃣ Model Training  
    6️⃣ Deployment  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>👥 Team</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    <p class='team-name'>🟦 Vanisha Arun</p>
    <p class='team-role'>ML Developer, UI/UX, Deployment</p>

    <p class='team-name'>🟩 Vaithiyanathan C</p>
    <p class='team-role'>Data Processing, Feature Engineering, Model Training</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🌐 Connect</p>", unsafe_allow_html=True)

    st.markdown("""
    <a class="icon-btn" href="https://github.com/VanishaarunV7" target="_blank">🐙 GitHub Vanisha</a>
    <a class="icon-btn" href="https://github.com/Vaithiy-777" target="_blank">🐙 GitHub Vaithiyanathan</a>
    <br><br>
    <a class="icon-btn" href="https://linkedin.com/in/vanishaarun7105" target="_blank">🔗 LinkedIn Vanisha</a>
    <a class="icon-btn" href="https://linkedin.com/in/vaithiy706" target="_blank">🔗 LinkedIn Vaithiyanathan</a>
    <br><br>
    <a class="icon-btn" href="mailto:vanisharuncse23@gmail.com" target="_blank">✉️ Mail Vanisha</a>
    <a class="icon-btn" href="mailto:vaiithiycm00@gmail.com" target="_blank">✉️ Mail Vaithiyanathan</a>
    """, unsafe_allow_html=True)

    st.markdown("<br><center style='color:#7dd3fc;'>✨ Created by Vanisha & Vaithiyanathan ✨</center>", unsafe_allow_html=True)
# =====================
# MAIN PAGE CONTENT
# =====================

# ⭐⭐ Paste here (Step 3 header) ⭐⭐
st.markdown("""
<h1 style='text-align:center; color:#2563eb; text-shadow:0 0 10px #bfdbfe;'>
🚢 Titanic Survival Predictor
</h1>
<p style='text-align:center; color:#475569;'>
Predict survival probability using Machine Learning
</p>
""", unsafe_allow_html=True)

# After this → your form fields start
pclass = st.selectbox("Passenger Class", [1,2,3])
gender = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age", min_value=0, max_value=100)
...

# ===================================
# LOAD MODEL + SCALER
# ===================================
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

# ===================================
# TITLES
# ===================================
st.markdown("<p class='big-title'>🚢 Titanic Survival Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict the chance of surviving the Titanic disaster</p>", unsafe_allow_html=True)

# ===================================
# INPUT SECTION
# ===================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎟 Passenger Class", [1, 2, 3])
    age = st.number_input("👶 Age", 1, 100, 25)
    sibsp = st.number_input("👨‍👩‍👦 Siblings/Spouses Aboard", 0, 10, 0)
    fare = st.number_input("💰 Fare Paid", 0.0, 600.0, 32.20)

with col2:
    sex = st.selectbox("⚥ Gender", ["male", "female"])
    parch = st.number_input("🧒 Parents/Children Aboard", 0, 10, 0)
    embarked = st.selectbox("🛳 Embarked From", ["C", "Q", "S"])

st.markdown("</div>", unsafe_allow_html=True)

# ===================================
# FEATURE ENGINEERING
# ===================================
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
familysize = sibsp + parch
isalone = 1 if familysize == 0 else 0

ordered_cols = [
    "Pclass","Age","SibSp","Parch","Fare",
    "FamilySize","IsAlone","Sex_male","Embarked_Q","Embarked_S"
]

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "FamilySize": familysize,
    "IsAlone": isalone,
    "Sex_male": sex_male,
    "Embarked_Q": embarked_Q,
    "Embarked_S": embarked_S
}])[ordered_cols]

scaled = scaler.transform(input_df.values)
input_df = pd.DataFrame(scaled, columns=ordered_cols)

# ===================================
# PREDICTION
# ===================================
if st.button("🔮 Predict Survival"):
    with st.spinner("Analyzing passenger details... please wait."):
        time.sleep(1.2)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    if pred == 1:
        st.success(f"🟢 SURVIVED — {prob:.2f}%")
        st.balloons()
    else:
        st.error(f"🔴 DID NOT SURVIVE — {prob:.2f}%")

# ===================================
# FOOTER
# ===================================
st.markdown(
    "<div class='footer'> Project by Vaithiyanathan C & Vanisha Arun</div>",
    unsafe_allow_html=True
)

# FLOATING TITANIC ICON
st.markdown("<div class='floating-icon'>🚢</div>", unsafe_allow_html=True)

# ======================
# ANIMATED BACKGROUND CSS
# ======================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: white;
    overflow-x: hidden;
}

/* Smooth Gradient Animation */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating particle bubbles */
.particle {
    position: fixed;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    animation: floatUp linear infinite;
}

@keyframes floatUp {
    0% { transform: translateY(0px); opacity: 0.2; }
    100% { transform: translateY(-900px); opacity: 0; }
}

/* Create random particle sizes + animation delays */
#particle1 { width: 12px; height: 12px; left: 10%; animation-duration: 16s; }
#particle2 { width: 8px; height: 8px; left: 25%; animation-duration: 12s; }
#particle3 { width: 15px; height: 15px; left: 40%; animation-duration: 14s; }
#particle4 { width: 10px; height: 10px; left: 55%; animation-duration: 18s; }
#particle5 { width: 20px; height: 20px; left: 70%; animation-duration: 22s; }
#particle6 { width: 7px; height: 7px; left: 85%; animation-duration: 20s; }

</style>

<!-- Particle elements -->
<div id="particle1" class="particle"></div>
<div id="particle2" class="particle"></div>
<div id="particle3" class="particle"></div>
<div id="particle4" class="particle"></div>
<div id="particle5" class="particle"></div>
<div id="particle6" class="particle"></div>

""", unsafe_allow_html=True)

