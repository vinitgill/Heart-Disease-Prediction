import streamlit as st
import numpy as np
import pickle

st.title("Heart Disease Prediction")

st.write("Enter patient details")

model = pickle.load(open("model.pkl", "rb"))

age = st.number_input("Age")

sex = st.selectbox("Sex", [0, 1])

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

restecg = st.selectbox("Rest ECG", [0, 1, 2])

thalach = st.number_input("Max Heart Rate")

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("Oldpeak")

slope = st.selectbox("Slope", [0, 1, 2])

ca = st.selectbox("CA", [0, 1, 2, 3, 4])

thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):

    features = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:

        st.error("Heart Disease Detected")

    else:

        st.success("No Heart Disease")