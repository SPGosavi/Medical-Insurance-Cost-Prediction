# <h1>Medical Insurance Cost Prediction</h1>

This project predicts the medical insurance cost based on various factors such as age, BMI, smoking status, and more. It uses machine learning (linear regression) for the prediction and is deployed using Streamlit for a user-friendly interface.

## <h2>Features</h2>
- Predicts insurance cost based on user input (age, BMI, sex, smoking status, etc.)
- Dynamic and interactive web app built with Streamlit
- Machine learning model trained using scikit-learn
- Real-time predictions and model evaluation

## <h2>Model and Algorithm</h2>
- **Algorithm**: Linear Regression  
- **Library**: scikit-learn  
The dataset used contains features like age, BMI, number of children, smoking status, and region to predict the insurance charges.

## <h2>Installation</h2>
To run this project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Medical_Insurance_Cost_Prediction.git
    cd Medical_Insurance_Cost_Prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run insurance_cost_app.py
    ```

## <h2>Dataset</h2>
The dataset used is publicly available at [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## <h2>How to Use</h2>
1. Open the deployed Streamlit app to try it live!  
   (Replace `#` with the actual link once deployed)
   
2. Enter details such as:
   - Age
   - BMI
   - Number of children
   - Gender (Male/Female)
   - Smoking status (Yes/No)
   - Region (Northeast, Northwest, Southeast, Southwest)

3. Click on **Predict Insurance Cost** to get the predicted amount.

## <h2>Model Evaluation</h2>
The model was evaluated using the **Mean Squared Error (MSE)** and performs reasonably well with unseen test data.

## <h2>Technologies Used</h2>
- **Python**: Data analysis and machine learning
- **scikit-learn**: Machine learning model
- **Pandas**: Data manipulation
- **Streamlit**: Web interface

## <h2>Streamlit App</h2>
Check out the live demo here: https://medical-insurance-cost-prediction-cbyny9ctquhdtbsflsshv6.streamlit.app/
