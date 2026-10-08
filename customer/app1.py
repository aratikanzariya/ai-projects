import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_tuned_model.pkl"

model = joblib.load(MODEL_PATH)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter customer details to predict whether the "
    "customer is likely to churn."
)

st.divider()


# =====================================================
# CUSTOMER DETAILS
# =====================================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


# =====================================================
# SERVICES
# =====================================================

st.subheader("📱 Services")

col1, col2 = st.columns(2)


with col1:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


with col2:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =====================================================
# BILLING
# =====================================================

st.subheader("💰 Billing")

col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=monthly_charges * tenure,
        step=10.0
    )


st.divider()


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    # Create customer DataFrame

    customer = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless_billing,

        "PaymentMethod": payment_method,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }])


    # =================================================
    # PREDICTION
    # =================================================

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    churn_probability = probability * 100

    no_churn_probability = 100 - churn_probability


    # =================================================
    # RESULT
    # =================================================

    st.divider()

    st.subheader("📈 Prediction Result")


    if prediction == 1:

        st.error("⚠️ CUSTOMER IS LIKELY TO CHURN")

    else:

        st.success("✅ CUSTOMER IS LIKELY TO STAY")


    # =================================================
    # PROBABILITY
    # =================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )


    with col2:

        st.metric(
            "No Churn Probability",
            f"{no_churn_probability:.2f}%"
        )


    # Progress bar

    st.write("**Churn Risk**")

    st.progress(
        int(churn_probability)
    )


    # =================================================
    # CUSTOMER SUMMARY
    # =================================================

    st.divider()

    st.subheader("📋 Customer Summary")

    st.write(
        f"**Tenure:** {tenure} months"
    )

    st.write(
        f"**Contract:** {contract}"
    )

    st.write(
        f"**Internet Service:** {internet_service}"
    )

    st.write(
        f"**Monthly Charges:** ${monthly_charges:.2f}"
    )

    st.write(
        f"**Total Charges:** ${total_charges:.2f}"
    )