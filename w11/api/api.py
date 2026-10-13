
from pathlib import Path
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"

# Load trained model and vectorizer
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

app = FastAPI(
    title="SMS Spam Detection API",
    description="Predict whether an SMS is spam",
    version="1.0"
)


class MessageRequest(BaseModel):
    message: str


def is_spam_label(label):
    """Recognize common spam labels."""
    value = str(label).strip().lower()
    return value in {"spam", "1", "true"}


@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: MessageRequest):
    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    try:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        result = (
            "Spam" if is_spam_label(prediction)
            else "Not Spam"
        )

        response_data = {
            "message": message,
            "prediction": result,
            "confidence": None
        }

        # Return probability when the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(message_vector)[0]
            classes = list(model.classes_)

            spam_indices = [
                index for index, label in enumerate(classes)
                if is_spam_label(label)
            ]

            if spam_indices:
                spam_probability = float(
                    probabilities[spam_indices[0]]
                )
                response_data["confidence"] = round(
                    spam_probability * 100, 2
                )

        return response_data

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )






# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
# VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


# with open(MODEL_PATH, "rb") as file:
#     model = pickle.load(file)


# with open(VECTORIZER_PATH, "rb") as file:
#     vectorizer = pickle.load(file)


# app = FastAPI(
#     title="SMS Spam Detection API",
#     description="API for detecting spam messages"
# )


# class MessageRequest(BaseModel):
#     message: str


# @app.get("/")
# def home():
#     return {
#         "message": "SMS Spam Detection API is running"
#     }


# @app.post("/predict")
# def predict(data: MessageRequest):

#     message = data.message

#     message_vector = vectorizer.transform([message])

#     prediction = model.predict(message_vector)[0]

#     print("Message:", message)
#     print("Prediction:", prediction)

#     if prediction == 1:
#         result = "Spam"
#     else:
#         result = "Not Spam"

#     return {
#         "message": message,
#         "prediction": result
#     }