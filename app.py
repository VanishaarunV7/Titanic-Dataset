import streamlit as st
import pandas as pd
import joblib
import time

# ===================================
# PAGE CONFIG
# ===================================
st.markdown("""
<h1 style='text-align:center; color:#2563eb; text-shadow:0 0 10px #bfdbfe;'>
🚢 Titanic Survival Predictor
</h1>
<p style='text-align:center; color:#475569; font-size:18px; margin-bottom:10px;'>
Predict survival probability using Machine Learning
</p>
""", unsafe_allow_html=True)

# ===================================
# LIGHT THEME + ANIMATED BACKGROUND
# ===================================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f6f9fc, #e9f3ff, #ffffff);
    background-size: 300% 300%;
    animation: moveBg 12s ease infinite;
    color: #1e293b;
    font-family: 'Segoe UI', sans-serif;
}

@keyframes moveBg {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

/* Floating blue bubbles */
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

/* Input box light mode */
input, select {
    background-color: white !important;
    color: #1e293b !important;
}

/* Card */
.card {
    padding: 20px;
    background: rgba(255,255,255,0.7);
    border-radius: 18px;
    border: 1px solid #d1d5db;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 25px;
}

/* Header */
.big-title {
    font-size: 42px;
    text-align: center;
    color: #2563eb;
    font-weight: bold;
    text-shadow: 0 0 10px #bfdbfe;
}

.sub-title {
    font-size: 18px;
    text-align: center;
    color: #475569;
    margin-bottom: 25px;
}

/* Floating ship */
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

<div id="p1" class="particle"></div>
<div id="p2" class="particle"></div>
<div id="p3" class="particle"></div>
<div id="p4" class="particle"></div>
<div id="p5" class="particle"></div>
<div id="p6" class="particle"></div>

""", unsafe_allow_html=True)

# ===================================
# SIDEBAR STYLING (LIGHT MODE)
# ===================================
st.markdown("""
<style>

/* SIDEBAR BACKGROUND */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #3b82f6, #1e40af, #0f172a);
    background-size: 300% 300%;
    animation: sidebarGlow 8s ease infinite;
    padding: 20px;
    color: white !important;
    border-right: 2px solid rgba(255,255,255,0.2);
}

/* Gradient animation */
@keyframes sidebarGlow {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

/* Make ALL text inside sidebar white */
.sidebar .sidebar-content * {
    color: white !important;
}

/* Section titles */
.section-title {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-top: 15px;
    color: #e0f2fe !important;
    text-shadow: 0 0 8px #93c5fd;
}

/* Main sidebar title */
.sidebar-title {
    font-size: 26px !important;
    font-weight: bold !important;
    margin-bottom: 10px;
    color: #ffffff !important;
    text-shadow: 0 0 15px #bae6fd;
}

/* Glass card blocks */
.glass-card {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 15px;
    backdrop-filter: blur(10px);
}

/* Team names */
.team-name {
    font-size: 18px;
    font-weight: bold;
    color: #bbf7d0 !important;
}

/* Team role text */
.team-role {
    font-size: 14px;
    color: #e0f2fe !important;
}

/* Social buttons */
.icon-btn {
    display: block;
    padding: 10px;
    border-radius: 10px;
    text-decoration: none;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    margin-top: 6px;
    text-align: center;
    transition: 0.3s;
    color: white !important;
}

.icon-btn:hover {
    background: rgba(255,255,255,0.35);
    transform: scale(1.05);
    box-shadow: 0 0 10px #93c5fd;
}

</style>
""", unsafe_allow_html=True)


# ===================================
# SIDEBAR CONTENT
# ===================================
with st.sidebar:

    st.markdown("<p class='sidebar-title'>🚢 Titanic ML Project</p>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>📘 About</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    A Machine Learning project predicting Titanic passenger survival 
    using Random Forest with a beautiful animated UI.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🛠 Tools Used</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    • Python<br>
    • Pandas<br>
    • Scikit-Learn<br>
    • Random Forest<br>
    • Streamlit<br>
    • GitHub<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>👥 Team</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
    <p class='team-name'>🟦 Vanisha Arun</p>
    <p class='team-role'>Data preprocessing, UI/UX, Deployment</p>
    <p class='team-name'>🟩 Vaithiyanathan C</p>
    <p class='team-role'>Ml developer,Train model,Feature Engineering</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='section-title'>🌐 Connect</p>", unsafe_allow_html=True)

    st.markdown("""
    <a class="icon-btn" href="https://github.com/VanishaarunV7" target="_blank">🐙 GitHub (Vanisha)</a><br>
    <a class="icon-btn" href="https://github.com/Vaithiy-777" target="_blank">🐙 GitHub (Vaithiyanathan)</a><br>
    <br>
    <a class="icon-btn" href="https://linkedin.com/in/vanishaarun7105" target="_blank">🔗 LinkedIn (Vanisha)</a><br>
    <a class="icon-btn" href="https://linkedin.com/in/vaithiy706" target="_blank">🔗 LinkedIn (Vaithiyanathan)</a><br>
    <br>
    <a class="icon-btn" href="mailto:vanisharuncse23@gmail.com">✉️ Mail Vanisha</a><br>
    <a class="icon-btn" href="mailto:vaiithiycm00@gmail.com">✉️ Mail Vaithiyanathan</a>
    """, unsafe_allow_html=True)

# ===================================
# MAIN PAGE HEADER
# ===================================
st.markdown("<p class='big-title'>🚢 Titanic Survival Predictor</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict survival probability using Machine Learning</p>", unsafe_allow_html=True)

# ===================================
# LOAD MODEL + SCALER
# ===================================
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

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
# FOOTER + FLOATING ICON
# ===================================
st.markdown(
    "<div class='sub-title'>✨ Project by Vaithiyanathan C & Vanisha Arun ✨</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='floating-icon'>🚢</div>", unsafe_allow_html=True)
