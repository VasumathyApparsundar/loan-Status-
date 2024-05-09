import streamlit as st # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore

# Load the model from the pickle file
model = joblib.load(r'C:/Users/Vasu/logistic_regression_model.pkl')

# Streamlit UI
st.title('Loan Approval Status Prediction')

# Define the prediction function
def predict_Status(features):
    features_array = [[features[feature] for feature in features]]
    prediction = model.predict(features_array)
    return prediction

# Define input fields for numerical variables
total_loan_amount = st.text_input("Total Loan Amount")
asset_value = st.text_input("Asset Value")
account_balance = st.text_input("Account Balance")
credit_score = st.text_input("Credit Score")
customer_income = st.text_input("Customer Income")
loan_term = st.text_input("Loan Term")
interest_rate = st.text_input("Interest Rate")

# Create a button to trigger the model prediction
if st.button('Predict'):
    
    # Get the values entered by the user
    input_values = {
        'total_loan_amount': float(total_loan_amount),
        'asset_value': float(asset_value),
        'account_balance': float(account_balance),
        'credit_score': float(credit_score),
        'customer_income': float(customer_income),
        'loan_term': float(loan_term),
        'interest_rate': float(interest_rate)
    }
    
    # Use the input values for prediction
    prediction = predict_Status(input_values)
    
    # Display the prediction
    if prediction[0] == 1:
        st.write("Prediction: The loan will not be Approved")
    else:
        st.write("Prediction: The loan will be Approved")

# Run the Streamlit app
st._main_()
