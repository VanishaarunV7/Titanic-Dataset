import streamlit as st
import pandas as pd
import joblib

# Load model + scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

st.title("🚢 Titanic Survival Prediction App")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", 1, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# One-hot encoding
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# New features
familysize = sibsp + parch
isalone = 1 if familysize == 0 else 0

# Input data (same as training columns)
input_data = pd.DataFrame([{
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
}])

# Scale numeric columns
num_cols = ['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone']
input_data[num_cols] = scaler.transform(input_data[num_cols])

if st.button("Predict"
