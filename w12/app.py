
import streamlit as st
import pickle
import os

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #c5c9d3;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #454b58;
        background-color: #1c222d;
    }

    .positive {
        color: #2ecc71;
        font-size: 32px;
        font-weight: bold;
    }

    .negative {
        color: #ff6464;
        font-size: 32px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        color: #9299a8;
        font-size: 13px;
        margin-top: 45px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 17px;
        font-weight: bold;
    }

    /* Review input: white background and black text */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        caret-color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
    }

    /* Placeholder text */
    .stTextArea textarea::placeholder {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }

    /* Review input label */
    .stTextArea label,
    .stTextArea label p {
        color: #ffffff !important;
    }

    /* Character count */
    .stTextArea + div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED MODEL ----------------
# Resolve the model path relative to this app.py file.
# This works when the Streamlit main file is w12/app.py.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}. "
            "Check that w12/models/sentiment_model.pkl "
            "exists in your GitHub repository."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎬 About the Project")

    st.write(
        "This application uses Machine Learning and Natural Language "
        "Processing (NLP) to identify the sentiment of movie reviews."
    )

    st.divider()

    st.subheader("✨ Features")
    st.write("• Positive / Negative prediction")
    st.write("• Class probability display")
    st.write("• Sample reviews")
    st.write("• Review text analysis")

    st.divider()

    st.subheader("🧠 Technology Used")
    st.write("• Python")
    st.write("• Scikit-learn")
    st.write("• TF-IDF Vectorization")
    st.write("• Logistic Regression")
    st.write("• Streamlit")

    st.caption("Week 12 | AI/ML Capstone Project")


# ---------------- MAIN HEADER ----------------
st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover whether a movie review expresses positive or negative feelings.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- INFORMATION CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎯 **Task**\n\nSentiment Classification")

with col2:
    st.info("⚙️ **Model**\n\nTF-IDF + Logistic Regression")

with col3:
    st.info("📊 **Output**\n\nPositive / Negative")


st.divider()


# ---------------- SAMPLE REVIEWS ----------------
st.subheader("📝 Enter Your Movie Review")

sample_reviews = {
    "Select a sample review": "",
    "Positive Review": (
        "This movie was absolutely fantastic! The acting was brilliant, "
        "the story was engaging, and I enjoyed every minute of it."
    ),
    "Negative Review": (
        "This movie was boring and disappointing. The story was weak, "
        "the acting was poor, and I did not enjoy watching it."
    )
}

selected_sample = st.selectbox(
    "Try an example review:",
    list(sample_reviews.keys())
)


# ---------------- REVIEW INPUT ----------------
review_input = st.text_area(
    "Write your review here:",
    value=sample_reviews[selected_sample],
    height=180,
    max_chars=10000,
    placeholder=(
        "Example: The movie was wonderful, "
        "with great acting and an amazing story..."
    ),
    key="review_input"
)

st.caption(f"Characters entered: {len(review_input)} / 10000")


# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Sentiment", type="primary"):

    if not review_input.strip():
        st.warning("Please enter a movie review before analyzing.")

    elif len(review_input.strip()) < 3:
        st.warning("Please enter a longer review.")

    else:
        try:
            # Predict sentiment using the saved ML pipeline
            prediction = model.predict([review_input])[0]
            prediction = str(prediction).strip().lower()

            st.divider()
            st.subheader("📊 Sentiment Analysis Result")

            # Display sentiment result
            if prediction == "positive":
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="positive">😊 Positive Review</div>
                        <p>The model classified this review as positive.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif prediction == "negative":
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="negative">😞 Negative Review</div>
                        <p>The model classified this review as negative.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.success(f"Predicted sentiment: {prediction.title()}")

            # Display probabilities if supported by the model
            if (
                hasattr(model, "predict_proba")
                and hasattr(model, "classes_")
            ):
                probabilities = model.predict_proba([review_input])[0]

                classes = [
                    str(label).strip().lower()
                    for label in model.classes_
                ]

                probability_map = dict(zip(classes, probabilities))

                positive_probability = float(
                    probability_map.get("positive", 0)
                )

                negative_probability = float(
                    probability_map.get("negative", 0)
                )

                st.markdown("### 📈 Prediction Probabilities")

                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.write("😊 Positive")
                    st.progress(positive_probability)
                    st.metric(
                        "Positive probability",
                        f"{positive_probability * 100:.2f}%"
                    )

                with prob_col2:
                    st.write("😞 Negative")
                    st.progress(negative_probability)
                    st.metric(
                        "Negative probability",
                        f"{negative_probability * 100:.2f}%"
                    )

                st.caption(
                    "Probabilities are model estimates, not a guarantee "
                    "that the predicted sentiment is correct."
                )

            # Display submitted review
            with st.expander("📄 View the review you submitted"):
                st.write(review_input)

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Movie Review Sentiment Analyzer | Machine Learning Capstone<br>
        Built with Python, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
















# import streamlit as st
# import pickle
# import os

# # ---------------- PAGE CONFIGURATION ----------------
# st.set_page_config(
#     page_title="Movie Review Sentiment Analyzer",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------------- CUSTOM CSS ----------------
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #0e1117;
#         color: #ffffff;
#     }

#     .main-title {
#         text-align: center;
#         font-size: 42px;
#         font-weight: 800;
#         color: #ff4b4b;
#         margin-bottom: 5px;
#     }

#     .subtitle {
#         text-align: center;
#         font-size: 17px;
#         color: #c5c9d3;
#         margin-bottom: 30px;
#     }

#     .result-card {
#         padding: 25px;
#         border-radius: 15px;
#         text-align: center;
#         margin-top: 20px;
#         border: 1px solid #454b58;
#         background-color: #1c222d;
#     }

#     .positive {
#         color: #2ecc71;
#         font-size: 32px;
#         font-weight: bold;
#     }

#     .negative {
#         color: #ff6464;
#         font-size: 32px;
#         font-weight: bold;
#     }

#     .footer {
#         text-align: center;
#         color: #9299a8;
#         font-size: 13px;
#         margin-top: 45px;
#     }

#     div.stButton > button {
#         width: 100%;
#         border-radius: 10px;
#         height: 48px;
#         font-size: 17px;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)


# # ---------------- LOAD TRAINED MODEL ----------------
# MODEL_PATH = os.path.join("models", "sentiment_model.pkl")

# @st.cache_resource
# def load_model():
#     with open(MODEL_PATH, "rb") as file:
#         return pickle.load(file)

# try:
#     model = load_model()
# except FileNotFoundError:
#     st.error(
#         "Model file not found! Please save your trained model at "
#         "models/sentiment_model.pkl"
#     )
#     st.stop()
# except Exception as error:
#     st.error(f"Unable to load model: {error}")
#     st.stop()


# # ---------------- SIDEBAR ----------------
# with st.sidebar:
#     st.title("🎬 About the Project")

#     st.write(
#         "This application uses Machine Learning and Natural Language "
#         "Processing (NLP) to identify the sentiment of movie reviews."
#     )

#     st.divider()

#     st.subheader("✨ Features")
#     st.write("• Positive / Negative prediction")
#     st.write("• Class probability display")
#     st.write("• Sample reviews")
#     st.write("• Review text analysis")

#     st.divider()

#     st.subheader("🧠 Technology Used")
#     st.write("• Python")
#     st.write("• Scikit-learn")
#     st.write("• TF-IDF Vectorization")
#     st.write("• Logistic Regression")
#     st.write("• Streamlit")

#     st.caption("Week 12 | AI/ML Capstone Project")


# # ---------------- MAIN HEADER ----------------
# st.markdown(
#     '<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     '<div class="subtitle">'
#     'Discover whether a movie review expresses positive or negative feelings.'
#     '</div>',
#     unsafe_allow_html=True
# )

# # ---------------- INFORMATION CARDS ----------------
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.info("🎯 **Task**\n\nSentiment Classification")

# with col2:
#     st.info("⚙️ **Model**\n\nTF-IDF + Logistic Regression")

# with col3:
#     st.info("📊 **Output**\n\nPositive / Negative")


# st.divider()

# # ---------------- SAMPLE REVIEWS ----------------
# st.subheader("📝 Enter Your Movie Review")

# sample_reviews = {
#     "Select a sample review": "",
#     "Positive Review": (
#         "This movie was absolutely fantastic! The acting was brilliant, "
#         "the story was engaging, and I enjoyed every minute of it."
#     ),
#     "Negative Review": (
#         "This movie was boring and disappointing. The story was weak, "
#         "the acting was poor, and I did not enjoy watching it."
#     )
# }

# selected_sample = st.selectbox(
#     "Try an example review:",
#     list(sample_reviews.keys())
# )

# # ---------------- REVIEW INPUT ----------------
# review_input = st.text_area(
#     "Write your review here:",
#     value=sample_reviews[selected_sample],
#     height=180,
#     max_chars=10000,
#     placeholder="Example: The movie was wonderful, with great acting and an amazing story..."
# )

# st.caption(f"Characters entered: {len(review_input)} / 10000")

# # ---------------- PREDICTION ----------------
# if st.button("🔍 Analyze Sentiment", type="primary"):

#     if not review_input.strip():
#         st.warning("Please enter a movie review before analyzing.")

#     elif len(review_input.strip()) < 3:
#         st.warning("Please enter a longer review.")

#     else:
#         try:
#             # Predict sentiment using the saved ML pipeline
#             prediction = model.predict([review_input])[0]
#             prediction = str(prediction).strip().lower()

#             # Show result
#             st.divider()
#             st.subheader("📊 Sentiment Analysis Result")

#             if prediction == "positive":
#                 st.markdown(
#                     """
#                     <div class="result-card">
#                         <div class="positive">😊 Positive Review</div>
#                         <p>The model classified this review as positive.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#             elif prediction == "negative":
#                 st.markdown(
#                     """
#                     <div class="result-card">
#                         <div class="negative">😞 Negative Review</div>
#                         <p>The model classified this review as negative.</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )

#             else:
#                 st.success(f"Predicted sentiment: {prediction.title()}")

#             # Display probabilities if the model supports them
#             if hasattr(model, "predict_proba"):
#                 probabilities = model.predict_proba([review_input])[0]
#                 classes = [
#                     str(label).strip().lower()
#                     for label in model.classes_
#                 ]

#                 probability_map = dict(zip(classes, probabilities))

#                 positive_probability = probability_map.get("positive", 0)
#                 negative_probability = probability_map.get("negative", 0)

#                 st.markdown("### 📈 Prediction Probabilities")

#                 prob_col1, prob_col2 = st.columns(2)

#                 with prob_col1:
#                     st.write("😊 Positive")
#                     st.progress(float(positive_probability))
#                     st.metric(
#                         "Positive probability",
#                         f"{positive_probability * 100:.2f}%"
#                     )

#                 with prob_col2:
#                     st.write("😞 Negative")
#                     st.progress(float(negative_probability))
#                     st.metric(
#                         "Negative probability",
#                         f"{negative_probability * 100:.2f}%"
#                     )

#                 st.caption(
#                     "Probabilities are model estimates, not a guarantee "
#                     "that the predicted sentiment is correct."
#                 )

#             # Display analyzed review
#             with st.expander("📄 View the review you submitted"):
#                 st.write(review_input)

#         except Exception as error:
#             st.error(f"Prediction failed: {error}")


# # ---------------- FOOTER ----------------
# st.markdown(
#     """
#     <div class="footer">
#         Movie Review Sentiment Analyzer | Machine Learning Capstone<br>
#         Built with Python, Scikit-learn and Streamlit
#     </div>
#     """,
#     unsafe_allow_html=True
# )