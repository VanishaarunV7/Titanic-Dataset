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

# ======================
# SUPER PROFESSIONAL SIDEBAR
# ======================
with st.sidebar:

    st.markdown("<h2 style='color:#38bdf8; font-weight:bold;'>📘 About This Project</h2>", unsafe_allow_html=True)
    st.write("""
This is a Machine Learning web application that predicts  
**Titanic Passenger Survival** using a **Random Forest Model**.  
Developed as a team project with a fully animated modern UI.
    """)

    st.markdown("---")

    # Tools Section
    st.markdown("<h3 style='color:#60a5fa;'>🛠 Tools & Technologies</h3>", unsafe_allow_html=True)
    st.write("""
- **Python 3.10**
- **Pandas** (Data Processing)  
- **Scikit-Learn** (Machine Learning)  
- **Random Forest Classifier**  
- **StandardScaler**  
- **Streamlit** (UI & Deployment)  
- **GitHub** (Version Control)  
- **Streamlit Cloud** (Hosting)  
    """)

    st.markdown("---")

    # ML Workflow
    st.markdown("<h3 style='color:#60a5fa;'>🔬 ML Workflow</h3>", unsafe_allow_html=True)
    st.write("""
1. Data Cleaning  
2. Feature Engineering  
3. One-Hot Encoding  
4. Scaling  
5. Model Training (Random Forest)  
6. Saving Model & Scaler  
7. Deploy via Streamlit  
    """)

    st.markdown("---")

    # Team Section
    st.markdown("<h3 style='color:#60a5fa;'>👥 Team Members</h3>", unsafe_allow_html=True)

    st.markdown("""
### 🟩 Vaithiyanathan C  
- ML Model Developer  
- Feature Engineering  
- Model Training & Optimization  
""")
    
    st.markdown("""
### 🟦 Vanisha Arun  
- Data pre processing  
- UI/UX & App Design  
- Deployment & GitHub  
""")

    st.markdown("---")

🐳 **Live App:**  
- Available through Streamlit Cloud  
""")

    st.markdown("---")

    st.markdown("<p style='text-align:center; color:#a5f3fc;'> Project by Vaithiyanathan C & Vanisha arun </p>", unsafe_allow_html=True)


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

