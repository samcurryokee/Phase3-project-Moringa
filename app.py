import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your trained model
model = joblib.load('XGB.joblib')

# Streamlit UI components
st.title('Customer Churn Prediction')
st.write("Adjust the input features to predict customer churn.")

# Input components for each feature

gender = st.radio("Gender", ["Male", "Female"])
senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])
partner = st.radio("Partner", ["No", "Yes"])
dependents = st.radio("Dependents", ["No", "Yes"])
multiple_lines = st.radio("Multiple Lines", ["No", "Yes"])
internet_service = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.radio(
    "Online Security", ["No", "Yes", "No internet service"])
online_backup = st.radio("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.radio(
    "Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.radio("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.radio("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.radio(
    "Streaming Movies", ["No", "Yes", "No internet service"])
paperless_billing = st.radio("Paperless Billing", ["No", "Yes"])
monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 100.0)
contract_one_year = st.radio("Contract One Year", ["No", "Yes"])
contract_two_year = st.radio("Contract Two Year", ["No", "Yes"])
average_monthly_charges = st.slider("Monthly Charges", 0.0, 10000.0, 100.0)

# Payment Method options
payment_method_options = ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"]
payment_method = st.selectbox("Payment Method", payment_method_options)

# Encode categorical features
label_encoder = LabelEncoder()
internet_service_encoded = label_encoder.fit_transform([internet_service])[0]
online_security_encoded = label_encoder.fit_transform([online_security])[0]
online_backup_encoded = label_encoder.fit_transform([online_backup])[0]
device_protection_encoded = label_encoder.fit_transform([device_protection])[0]
tech_support_encoded = label_encoder.fit_transform([tech_support])[0]
streaming_tv_encoded = label_encoder.fit_transform([streaming_tv])[0]
streaming_movies_encoded = label_encoder.fit_transform([streaming_movies])[0]

# Collect input features in a dictionary
input_data = {
    'gender': 0 if gender == "Male" else 1,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'MultipleLines': 1 if multiple_lines == "Yes" else 0,
    'InternetService': internet_service_encoded,
    'OnlineSecurity': online_security_encoded,
    'OnlineBackup': online_backup_encoded,
    'DeviceProtection': device_protection_encoded,
    'TechSupport': tech_support_encoded,
    'StreamingTV': streaming_tv_encoded,
    'StreamingMovies': streaming_movies_encoded,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'MonthlyCharges': monthly_charges,
    'Contract_One year': 1 if contract_one_year == "Yes" else 0,
    'Contract_Two year': 1 if contract_two_year == "Yes" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    'AvgMonthlyCharges': average_monthly_charges,
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Make predictions
prediction = model.predict(input_df)[0]

# Display the result
if prediction == 0:
    st.write("Prediction: The customer is likely to stay.")
else:
    st.write("Prediction: The customer is likely to churn.")
