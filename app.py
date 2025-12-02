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
.sidebar .sidebar-content {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(200,200,200,0.5);
}

.sidebar-title {
    color:#2563eb;
    font-size: 26px;
    font-weight:bold;
}

.section-title {
    color:#3b82f6;
    font-size:20px;
    margin-top:15px;
}

.glass-card {
    background: rgba(255,255,255,0.7);
    border:1px solid rgba(200,200,200,0.6);
    border-radius:12px;
    padding:12px;
    margin-bottom:15px;
}

.icon-btn {
    display:inline-block;
    padding:8px 14px;
    margin:6px 0;
    border-radius:10px;
    background:white;
    border:1px solid #cbd5e1;
    text-decoration:none;
    color:#1e293b;
}
.icon-btn:hover {
    background:#e0f2fe;
}
.team-name { font-weight:bold; color:#2563eb; }
.team-role { color:#475569; font-size:14px; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.sidebar .sidebar-content * {
    color: white !important;
}
.section-title {
    color: #93c5fd !important;
}
.team-name {
    color: #7dd3fc !important;
}
.team-role {
    color: #cbd5e1 !important;
}
.glass-card {
    color: white !important;
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
