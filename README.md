Medical Insurance Cost Prediction
This project predicts the medical insurance cost based on various factors such as age, BMI, smoking status, and more. It uses machine learning (linear regression) for the prediction and is deployed using Streamlit for a user-friendly interface.

Features
Predicts insurance cost based on user input (age, BMI, sex, smoking status, etc.)
Dynamic and interactive web app built with Streamlit
Machine learning model trained using scikit-learn
Real-time predictions and model evaluation
Model and Algorithm
Algorithm: Linear Regression
Library: scikit-learn
The dataset used contains features like age, BMI, number of children, smoking status, and region to predict the insurance charges.

Installation
To run this project locally:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/Medical_Insurance_Cost_Prediction.git
cd Medical_Insurance_Cost_Prediction
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run insurance_cost_app.py
Dataset
The dataset used is publicly available at Medical Cost Personal Dataset.

How to Use
Open the deployed Streamlit app to try it live! (Replace # with the actual link once deployed)
Enter details such as:
Age
BMI
Number of children
Gender (Male/Female)
Smoking status (Yes/No)
Region (Northeast, Northwest, Southeast, Southwest)
Click on Predict Insurance Cost to get the predicted amount.
Model Evaluation
The model was evaluated using the Mean Squared Error (MSE) and performs reasonably well with unseen test data.

Technologies Used
Python: Data analysis and machine learning
scikit-learn: Machine learning model
Pandas: Data manipulation
Streamlit: Web interface
Streamlit App
