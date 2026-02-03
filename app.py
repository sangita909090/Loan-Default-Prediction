import streamlit as st
import joblib
import numpy as np

# Load the trained model (ensure loan_model.pkl is in the same folder)
model = joblib.load('loan_model.pkl')

st.title("🏦 Loan Default Predictor")
st.write("Enter applicant details to predict default risk.")
st.divider()

# Input fields with tooltips and validation
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        help="Age of the applicant (18–100 years)"
    )

with col2:
    income = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Gross annual income in dollars"
    )

credit_score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600,
    help="FICO credit score (300–850). Higher is better."
)

col3, col4 = st.columns(2)

with col3:
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=0,
        help="Number of financial dependents (e.g., spouse, children)"
    )

with col4:
    home_owner = st.selectbox(
        "Home Owner",
        options=("No", "Yes"),
        help="Whether the applicant owns their home"
    )

home_owner_val = 1 if home_owner == "Yes" else 0

st.divider()

# Validation and prediction
if st.button("Run our model", use_container_width=True):
    # Validate inputs
    if income < 1000:
        st.warning("⚠️ Annual income seems very low. Please verify.")
    
    if credit_score < 500:
        st.warning("⚠️ Credit score < 500 || high risk indicator.")
    
    features = np.array([[age, income, credit_score, dependents, home_owner_val]])
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.divider()
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Default**")
        st.write(f"Risk probability: **{prediction_proba[0][1]:.1%}**")
        st.info("This applicant is likely to default. Consider additional verification or declining the loan.")
    else:
        st.success("✅ **Safe to Approve**")
        st.write(f"Default probability: **{prediction_proba[0][1]:.1%}**")
        st.info("This applicant shows low risk of default.")
