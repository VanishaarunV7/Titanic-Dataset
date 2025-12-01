import streamlit as st
import pandas as pd
import joblib

# Load model + scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("titanic_scaler.pkl")

st.title("🚢 Titanic Survival Prediction App")

# User input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encoding categorical features
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Feature engineering
familysize = sibsp + parch
isalone = 1 if familysize == 0 else 0

# Construct input in EXACT training column order
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

# Ensure correct column order
ordered_columns = [
    'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S'
]

input_data = input_data[ordered_columns]

# Scale numeric columns
num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# Prediction
if st.button("Predict"):
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.success("🟢 Passenger Survived!")
    else:
        st.error("🔴 Passenger Did NOT Survive.")
