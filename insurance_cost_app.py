import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('insurance.csv')

# Convert categorical variables into numeric
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].astype('category').cat.codes

# Features and target
X = data[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
y = data['charges']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Medical Insurance Cost Prediction")

# Input fields for user data
age = st.slider("Age", 18, 100)
bmi = st.slider("BMI", 15, 50)
children = st.slider("Number of Children", 0, 5)
sex = st.selectbox("Sex", ['Male', 'Female'])
smoker = st.selectbox("Smoker", ['Yes', 'No'])
region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

# Encode user input
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0
region = ['Northeast', 'Northwest', 'Southeast', 'Southwest'].index(region)

# Predict button
if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, bmi, children, sex, smoker, region]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Insurance Cost: ${prediction[0]:,.2f}")

# Model evaluation
mse = mean_squared_error(y_test, model.predict(X_test))
st.write(f"Mean Squared Error on Test Set: {mse:.2f}")
