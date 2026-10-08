import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

model = joblib.load(MODEL_PATH)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# =====================================================
# TITLE
# =====================================================

st.title("🚗 Used Car Price Predictor")

st.write(
    "Enter your car details below to predict its estimated "
    "selling price."
)

st.divider()


# =====================================================
# CAR INFORMATION
# =====================================================

st.subheader("🚘 Enter Car Details")


# Car Name
car_name = st.text_input(
    "Car Name",
    placeholder="Example: city"
)


# Two columns
col1, col2 = st.columns(2)


with col1:

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.1,
        value=5.0,
        step=0.1
    )

    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=25000,
        step=1000
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "CNG"
        ]
    )

    seller_type = st.selectbox(
        "Seller Type",
        [
            "Dealer",
            "Individual"
        ]
    )


with col2:

    transmission = st.selectbox(
        "Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )

    owner = st.selectbox(
        "Previous Owners",
        [
            0,
            1,
            2,
            3
        ]
    )

    car_age = st.number_input(
        "Car Age (Years)",
        min_value=0,
        value=4,
        step=1
    )


# =====================================================
# CALCULATE KMS PER YEAR
# =====================================================

kms_per_year = kms_driven / max(car_age, 1)


st.divider()


# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮 Predict Selling Price",
    use_container_width=True
):

    # Check car name
    if car_name.strip() == "":

        st.warning("⚠️ Please enter the car name.")

    else:

        # =============================================
        # CREATE NEW CAR DATA
        # =============================================

        new_car = pd.DataFrame([{

            "Car_Name": car_name.strip(),

            "Present_Price": present_price,

            "Kms_Driven": kms_driven,

            "Fuel_Type": fuel_type,

            "Seller_Type": seller_type,

            "Transmission": transmission,

            "Owner": owner,

            "Car_Age": car_age,

            "Kms_Per_Year": kms_per_year

        }])


        # =============================================
        # PREDICT
        # =============================================

        predicted_price = model.predict(new_car)[0]


        # =============================================
        # DISPLAY RESULT
        # =============================================

        st.success("✅ Prediction Completed!")

        st.subheader("💰 Estimated Selling Price")


        st.metric(
            "Predicted Price",
            f"₹ {predicted_price:.2f} Lakhs"
        )


        st.write(
            f"### Approximately ₹ {predicted_price * 100000:,.0f}"
        )


        # =============================================
        # CAR DETAILS
        # =============================================

        st.divider()

        st.subheader("📋 Car Details")


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                f"**Car:** {car_name.title()}"
            )

            st.write(
                f"**Present Price:** ₹{present_price:.2f} Lakhs"
            )

            st.write(
                f"**Kilometers:** {kms_driven:,} km"
            )


        with col2:

            st.write(
                f"**Fuel:** {fuel_type}"
            )

            st.write(
                f"**Transmission:** {transmission}"
            )

            st.write(
                f"**Previous Owners:** {owner}"
            )


        st.write(
            f"**Car Age:** {car_age} years"
        )

        st.write(
            f"**Kilometers per Year:** {kms_per_year:,.2f} km"
        )