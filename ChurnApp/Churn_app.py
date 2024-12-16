import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# Define prediction function
def make_prediction(
        gender, SeniorCitizen, Partner, Dependents, tenure, Phoneservice,
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
):
    # Make a dataframe from input data
    input_data = pd.DataFrame({
        'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
        'Dependents': [Dependents], 'tenure': [tenure], 'PhoneService': [Phoneservice],
        'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
        'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod], 'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })
    
    
     # Load already saved pipeline and make predictions
    preprocessor_filepath = os.path.join(os.getcwd(), 'models', 'preprocessor.joblib')
    if os.path.exists(preprocessor_filepath):
        with open(preprocessor_filepath, "rb") as p:
            preprocessor = joblib.load(p)
            input_data = preprocessor.transform(input_data)
    else:
        st.error("Error: preprocessor.joblib file not found.")
    predt = None
    # Load already saved pipeline and make predictions
    rf_model_filepath = os.path.join(os.getcwd(), 'models', 'rf_model.joblib')
    if os.path.exists(rf_model_filepath):
        with open(rf_model_filepath, "rb") as f:
            model = joblib.load(f)
            predt = model.predict(input_data)
    else:
        st.error("Error: rf_model.joblib file not found.")

    # Return prediction
    if np.any(predt == 1):
        return 'Customer Will Churn'
    else:
        return 'Customer Will Not Churn'

# Create the input components for Streamlit with helps
gender_input = st.selectbox('Select gender', ['Female', 'Male'], help='Select gender')
SeniorCitizen_input = st.selectbox('Is the customer a senior citizen?', ['Yes', 'No'], help='Is the customer a senior citizen?')
Partner_input = st.selectbox('Has the customer a partner?', ['Yes', 'No'], help='Has the customer a partner?')
Dependents_input = st.selectbox('Does the customer have dependents?', ['Yes', 'No'], help='Does the customer have dependents?')
tenure_input = st.number_input('Number of months the customer has stayed with the company', help='Enter a number')
Phoneservice_input = st.selectbox('Does the customer have phone service?', ['Yes', 'No'], help='Does the customer have phone service?')
MultipleLines_input = st.selectbox('Does the customer have multiple phone lines?', ['No phone service', 'No', 'Yes'], help='Does the customer have multiple phone lines?')
InternetService_input = st.selectbox('Type of Internet service', ['DSL', 'Fiber optic', 'No'], help='Type of Internet service')
OnlineSecurity_input = st.selectbox('Does the customer have online security?', ['No', 'Yes', 'No internet service'], help='Does the customer have online security?')
OnlineBackup_input = st.selectbox('Does the customer have online backup?', ['Yes', 'No', 'No internet service'], help='Does the customer have online backup?')
DeviceProtection_input = st.selectbox('Does the customer have device protection?', ['No', 'Yes', 'No internet service'], help='Does the customer have device protection?')
TechSupport_input = st.selectbox('Does the customer have tech support?', ['No', 'Yes', 'No internet service'], help='Does the customer have tech support?')
StreamingTV_input = st.selectbox('Does the customer have streaming TV?', ['No', 'Yes', 'No internet service'], help='Does the customer have streaming TV?')
StreamingMovies_input = st.selectbox('Does the customer have streaming movies?', ['No', 'Yes', 'No internet service'], help='Does the customer have streaming movies?')
Contract_input = st.selectbox('Type of contract', ['Month-to-month', 'One year', 'Two year'], help='Type of contract')
PaperlessBilling_input = st.selectbox('Is the customer using paperless billing?', ['Yes', 'No'], help='Is the customer using paperless billing?')
PaymentMethod_input = st.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], help='Payment method')
MonthlyCharges_input = st.number_input('Monthly Charges', help='Enter the Monthly charges')
TotalCharges_input = st.number_input('Total Charges', help='Enter the Total charges')


# Set the title of the Streamlit app
st.title('Customer Churn Prediction App')

# Use the inputs in your app
if st.button('Make Prediction'):
    prediction_result = make_prediction(
        gender_input, SeniorCitizen_input, Partner_input, Dependents_input, tenure_input,
        Phoneservice_input, MultipleLines_input, InternetService_input, OnlineSecurity_input,
        OnlineBackup_input, DeviceProtection_input, TechSupport_input, StreamingTV_input,
        StreamingMovies_input, Contract_input, PaperlessBilling_input, PaymentMethod_input,
        MonthlyCharges_input, TotalCharges_input
    )
    st.write("Prediction Result:", prediction_result)