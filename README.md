# 🚢 Titanic Survival Prediction App

A machine-learning powered Titanic Survival Prediction Web App built using Python, Streamlit, and Scikit-learn.
The app predicts whether a passenger would have survived the Titanic disaster based on key features like age, gender, ticket class, family size, and more.

🔗 Live App:
👉 https://titanic-dataset-cig2htmxlyqgpnm3axqpyj.streamlit.app/

⭐ Project Overview

This project uses the famous Titanic Dataset to train a Random Forest Classifier that predicts the survival chance of a passenger.

The web interface is developed using Streamlit, featuring:

Modern UI

Gradient background

Animated icons

Sidebar info

Live survival predictions

Interactive form inputs

Probabilities & visual feedback

⚙️ Tech Stack
Component	Technology
UI / Web App	Streamlit
ML Algorithm	Random Forest
Language	Python
Data Processing	Pandas, Scikit-learn
Deployment	Streamlit Cloud
Version Control	GitHub
📂 Dataset

Titanic-Dataset.csv

Contains information about ~900 passengers

Columns include:

Pclass

Age

Sex

Fare

Siblings/Spouses (SibSp)

Parents/Children (Parch)

Embarked

Survived

🧠 Machine Learning Workflow

Load dataset

Clean missing values

Feature engineering

FamilySize

IsAlone

One-hot encoding

Scaling with StandardScaler

Train Random Forest Classifier

Save:

titanic_model.pkl

titanic_scaler.pkl

Deploy app with Streamlit Cloud

🎨 App Features

Premium modern UI

Floating Titanic icon

Animated footer

Dark theme gradient

Card-style inputs

Balloons on successful prediction

Probabilistic prediction (%)

Easy-to-use interface

🖥 How to Run Locally
1. Clone repo
git clone https://github.com/VanishaarunV7/Titanic-Dataset.git
cd Titanic-Dataset

2. Install required libraries
pip install -r requirements.txt

3. Run Streamlit
streamlit run app.py

🧑‍🤝‍🧑 Team Members

This project was built by:

👩‍💻 Vanisha Arun
🧑‍💻 Vaithiyanathan C

A combined effort to develop a clean & professional ML application.

🚀 Deployment

The app is deployed using Streamlit Cloud, allowing direct access without tunnels, servers, or local hosting.

❤️ Acknowledgements

Special thanks to the open-source community, Streamlit, and Scikit-learn for providing amazing tools to bring ML apps to life.

🏁 Final Note

This project showcases how Machine Learning can be applied to real-world historical datasets, delivering insights in an interactive and user-friendly way.
