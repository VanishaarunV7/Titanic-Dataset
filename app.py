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

# ===================================
# SIDEBAR
# ===================================
with st.sidebar:
    st.header("About This Project ✨")
    st.write("""
    🚢 **Titanic Survival Prediction App**  
    Built using **Machine Learning + Streamlit**.

    **Team:**  
    🔹 Vanisha Arun  
    🔹 Vaithiyanathan C

    **Features:**  
    ✔ Clean UI  
    ✔ Modern design  
    ✔ Live prediction  
    ✔ Random Forest ML Model  
    """)

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
    "<div class='footer'>✨ Project by Vanisha & Vaithiyanathan ✨</div>",
    unsafe_allow_html=True
)

# FLOATING TITANIC ICON
st.markdown("<div class='floating-icon'>🚢</div>", unsafe_allow_html=True)
