# Importing the Dependencies
import numpy as np
import pandas as pd
import pickle
import streamlit as st


from src.mlproject.utils import load_object

# Load the preprocessor and model
preprocessor = load_object("artifacts/preprocessor.pkl")
model = load_object("artifacts/model.pkl")

st.title("Student Score Predictor (Math)")

# Input fields for user
gender = st.selectbox("Gender", ["female", "male"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Create DataFrame
input_df = pd.DataFrame(
    {
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_level_of_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score],
    }
)

# Predict on button click
if st.button("Predict Math Score"):
    data_scaled = preprocessor.transform(input_df)
    prediction = model.predict(data_scaled)
    st.success(f"Predicted Math Score: {round(prediction[0], 2)}")
