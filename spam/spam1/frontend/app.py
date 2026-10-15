import streamlit as st
import requests


# st.set_page_config(
#     page_title="SMS Spam Detector",
#     page_icon="📩"
# )


st.title("📩 SMS Spam Detector")

st.write(
    "Enter an SMS message and check whether it is Spam or Not Spam."
)


message = st.text_area(
    "Enter your message:",
    height=150
)


if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"message": message},
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]

                if prediction == "Spam":
                    st.error("🚨 This message is SPAM!")

                else:
                    st.success("✅ This message is NOT SPAM.")

            else:
                st.error("API returned an error.")

        except requests.exceptions.RequestException:
            st.error(
                "Could not connect to FastAPI. "
                "Please make sure the API is running."
            )