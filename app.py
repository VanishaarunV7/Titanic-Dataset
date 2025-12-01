import streamlit as st
import pandas as pd
import joblib
import time

# ====================
# PAGE CONFIG
# ====================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ====================
# CUSTOM CSS
# ====================
st.markdown("""
<style>
body {
    background: #0d1117;
    color: white;
}
.big-title {
    font-size: 40px;
    text-align: center;
    color: #4db8ff;
    font-weight: bold;
}
.sub-title {
    font-size: 20px;
    text-align: center;
    color: #b3e0ff;
    margin-bottom: 20px;
}
.card {
    padding: 20px;
    background-color: #1b2431;
    border-radius: 15px;
    box-shadow: 0px 0px 10px #334155;
}
</style>
""", unsafe_allow_html=True)

# ====================
# SIDEBAR
# ====================
with st.sidebar:
    st.image("https://i.imgur.com/hsYVx4T.png", use_column_width=True)
    st.header("About App")
    st.write("""
    🚢 This app predicts whether a passenger  
    would have **survived the Titanic disaster**  
    using Machine Learning.

    **Created by:** Vanisha  
    **Model:** Random Forest  
    """)

# ====================
# LOAD MODEL
# ====================
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

# ====================
# TITLE
# ====================
st.markdown("<p class='big-title'>🚢 Titanic Survival Prediction</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter passenger details to know the survival chance</p>", unsafe_allow_html=True)

# ====================
# INPUT FORM
# ====================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("🎟 Passenger Class", [1, 2, 3])
        age = st.number_input("👶 Age", 1, 100, 25)
        sibsp = st.number_input("👨‍👩‍👦 Siblings/Spouses", 0, 10, 0)
        fare = st.number_input("💰 Fare Paid", 0.0, 600.0, 32.2)

    with col2:
        sex = st.selectbox("⚥ Gender", ["male", "female"])
        parch = st.number_input("🧒 Parents/Children", 0, 10, 0)
        embarked = st.selectbox("🛳 Port of Embarkation", ["C", "Q", "S"])

    st.markdown("</div>", unsafe_allow_html=True)

# ====================
# FEATURE ENGINEERING
# ====================
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0
familysize = sibsp + parch
isalone = 1 if familysize == 0 else 0

ordered_cols = [
    'Pclass','Age','SibSp','Parch','Fare',
    'FamilySize','IsAlone','Sex_male','Embarked_Q','Embarked_S'
]

input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'FamilySize': familysize,
    'IsAlone': isalone,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}])[ordered_cols]

scaled = scaler.transform(input_df.values)
input_df = pd.DataFrame(scaled, columns=ordered_cols)

# ====================
# PREDICTION
# ====================
if st.button("🔮 Predict Survival"):
    with st.spinner("Analyzing passenger details..."):
        time.sleep(1.5)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] * 100

    if pred == 1:
        st.success(f"🟢 SURVIVED — {proba:.2f}% chance")
        st.balloons()
    else:
        st.error(f"🔴 DID NOT SURVIVE — {proba:.2f}% chance")

