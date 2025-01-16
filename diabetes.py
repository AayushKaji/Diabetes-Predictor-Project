import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("C:/Users/AAYUSH/Desktop/Aayush Kaji/miniproject/diabetes.csv")

# Split data into features and target
x = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

# Standardize the dataset
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter your medical details below to predict if you are diabetic.")

# User input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%f")
age = st.number_input("Age", min_value=1, max_value=120, step=1)

# Prediction button
if st.button("Predict"):
    # Prepare input data
    user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])
    user_input_reshaped = user_input.reshape(1, -1)
    std_data = scaler.transform(user_input_reshaped)

    # Make prediction
    prediction = classifier.predict(std_data)

    # Display result
    if prediction[0] == 0:
        st.success("The person is Non-Diabetic.")
    else:
        st.error("The person is Diabetic.")
