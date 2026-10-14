# import streamlit as st
# import requests


# # st.set_page_config(
# #     page_title="SMS Spam Detector",
# #     page_icon="📩"
# # )


# st.title("📩 SMS Spam Detector")

# st.write(
#     "Enter an SMS message and check whether it is Spam or Not Spam."
# )


# message = st.text_area(
#     "Enter your message:",
#     height=150
# )


# if st.button("Predict"):

#     if message.strip() == "":
#         st.warning("Please enter a message.")

#     else:

#         try:

#             response = requests.post(
#                 "http://127.0.0.1:8000/predict",
#                 json={"message": message},
#                 timeout=10
#             )

#             if response.status_code == 200:

#                 result = response.json()

#                 prediction = result["prediction"]

#                 if prediction == "Spam":
#                     st.error("🚨 This message is SPAM!")

#                 else:
#                     st.success("✅ This message is NOT SPAM.")

#             else:
#                 st.error("API returned an error.")

#         except requests.exceptions.RequestException:
#             st.error(
#                 "Could not connect to FastAPI. "
#                 "Please make sure the API is running."
#             )


import streamlit as st
import pickle
from pathlib import Path


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


# -----------------------------
# Load Model and Vectorizer
# -----------------------------
@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# -----------------------------
# UI
# -----------------------------
st.title("📩 SMS Spam Detector")

st.write(
    "Enter an SMS message and check whether it is Spam or Not Spam."
)


message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Example: Congratulations! You won a free prize!"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict", type="primary"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is NOT SPAM.")