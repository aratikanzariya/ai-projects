from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.pkl"


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


app = FastAPI(
    title="SMS Spam Detection API",
    description="API for detecting spam messages"
)


class MessageRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "SMS Spam Detection API is running"
    }


@app.post("/predict")
def predict(data: MessageRequest):

    message = data.message

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return {
        "message": message,
        "prediction": result
    }