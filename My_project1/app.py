import streamlit as st
import joblib
from pathlib import Path

# ==========================================
# LOAD MODEL AND TF-IDF VECTORIZER
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)

st.title("📱 SMS Spam Detector")

st.write(
    "Enter an SMS message below to check whether it is "
    "Ham (Normal) or Spam."
)

st.divider()


# ==========================================
# MESSAGE INPUT
# ==========================================

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize!"
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")

    else:

        # Convert message into TF-IDF
        message_tfidf = tfidf.transform([message])

        # Prediction
        prediction = model.predict(message_tfidf)[0]

        # Probability
        probability = model.predict_proba(message_tfidf)[0]

        # ==========================================
        # LABEL MAPPING
        # 0 = Ham
        # 1 = Spam
        # ==========================================

        if prediction == 0:

            result = "Ham"

            ham_probability = probability[0] * 100
            spam_probability = probability[1] * 100

            st.success("✅ HAM MESSAGE")

        else:

            result = "Spam"

            ham_probability = probability[0] * 100
            spam_probability = probability[1] * 100

            st.error("🚨 SPAM MESSAGE")


        # ==========================================
        # SHOW RESULT
        # ==========================================

        st.subheader("Prediction")

        st.write(f"**Result:** {result}")


        # ==========================================
        # PROBABILITY
        # ==========================================

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Ham Probability",
                f"{ham_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )


        # ==========================================
        # PROGRESS BARS
        # ==========================================

        st.write("**Ham Probability**")
        st.progress(int(ham_probability))

        st.write("**Spam Probability**")
        st.progress(int(spam_probability))


        # ==========================================
        # ORIGINAL MESSAGE
        # ==========================================

        st.divider()

        st.write("**Original Message:**")
        st.info(message)