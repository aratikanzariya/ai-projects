
import os
import requests
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide"
)

# Backend URL
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("📩 SMS Spam Detection System")
st.caption("Machine Learning powered message classification")

st.markdown(
    "Check whether an SMS message is **Spam** or **Not Spam**."
)

with st.sidebar:
    st.header("About Project")
    st.write("Model: Trained Machine Learning classifier")
    st.write("Input: SMS text")
    st.write("Output: Spam / Not Spam")
    st.divider()
    st.subheader("Try an example")

    if st.button("🎁 Prize message"):
        st.session_state.example = (
            "Congratulations! You have won a free prize. Claim now!"
        )

    if st.button("👋 Normal message"):
        st.session_state.example = (
            "Hi, are we meeting for lunch today?"
        )

    if st.button("📅 Reminder message"):
        st.session_state.example = (
            "Reminder: Your appointment is tomorrow at 10 AM."
        )

    if st.button("🧹 Clear history"):
        st.session_state.history = []
        st.rerun()

if "example" not in st.session_state:
    st.session_state.example = ""

st.subheader("🔍 Check a message")

message = st.text_area(
    "Enter your SMS message",
    value=st.session_state.example,
    height=130,
    placeholder="Type or paste an SMS message here..."
)

if st.button("🚀 Predict Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        try:
            with st.spinner("Analyzing your message..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"message": message.strip()},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

            prediction = data.get("prediction", "Unknown")
            confidence = data.get("confidence")

            if prediction == "Spam":
                st.error("🚨 This message is SPAM!")
            elif prediction == "Not Spam":
                st.success("✅ This message is NOT SPAM.")
            else:
                st.warning(f"Prediction: {prediction}")

            if confidence is not None:
                st.metric(
                    "Model confidence",
                    f"{float(confidence):.2f}%"
                )
                st.caption(
                    "Confidence is a model estimate, not a guarantee."
                )

            st.session_state.history.insert(
                0,
                {
                    "Message": message.strip(),
                    "Prediction": prediction,
                    "Confidence (%)": (
                        round(float(confidence), 2)
                        if confidence is not None else "N/A"
                    )
                }
            )

        except requests.exceptions.RequestException as error:
            st.error(
                "Could not connect to the backend. "
                "Check that FastAPI is running and API_URL is correct."
            )
            st.caption(str(error))
        except (ValueError, TypeError, KeyError):
            st.error("The API returned an invalid prediction response.")

st.divider()

# Multiple message testing
st.subheader("📚 Test Multiple Messages")
st.write("Enter one SMS per line.")

bulk_text = st.text_area(
    "Multiple SMS messages",
    height=120,
    placeholder=(
        "Congratulations! You won a prize.\n"
        "Hi, see you tomorrow."
    ),
    key="bulk_messages"
)

if st.button("Analyze All Messages"):
    messages = [
        line.strip()
        for line in bulk_text.splitlines()
        if line.strip()
    ]

    if not messages:
        st.warning("Enter at least one message.")
    else:
        results = []

        progress = st.progress(0)

        for index, sms in enumerate(messages):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"message": sms},
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()

                results.append({
                    "Message": sms,
                    "Prediction": result.get("prediction", "Unknown"),
                    "Confidence (%)": (
                        round(float(result["confidence"]), 2)
                        if result.get("confidence") is not None
                        else "N/A"
                    )
                })

            except (requests.exceptions.RequestException,
                    ValueError, TypeError):
                results.append({
                    "Message": sms,
                    "Prediction": "Error",
                    "Confidence (%)": "N/A"
                })

            progress.progress((index + 1) / len(messages))

        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download results as CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
            file_name="spam_detection_results.csv",
            mime="text/csv"
        )

# History
st.divider()
st.subheader("🕘 Prediction History")

if st.session_state.history:
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Your predictions will appear here during this session.")
