import streamlit as st
import keras
import numpy as np
from PIL import Image
from pathlib import Path

# --------------------------------
# Load Model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_model.keras"

model = keras.models.load_model(MODEL_PATH)

# --------------------------------
# Page
# --------------------------------

st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢"
)

st.title("🔢 MNIST Handwritten Digit Predictor")

st.write("Upload a handwritten digit image.")

# --------------------------------
# Upload
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Original Image")
    st.image(image, width=200)

    # --------------------------------
    # Resize
    # --------------------------------

    image = image.resize((28, 28))

    # --------------------------------
    # Convert to numpy
    # --------------------------------

    image_array = np.array(image)

    # --------------------------------
    # Invert if background is white
    # --------------------------------

    if image_array.mean() > 127:
        image_array = 255 - image_array

    # --------------------------------
    # Normalize
    # --------------------------------

    image_array = image_array.astype("float32") / 255.0

    # --------------------------------
    # Add batch dimension
    # --------------------------------

    image_array = np.expand_dims(image_array, axis=0)

    # --------------------------------
    # Prediction
    # --------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = np.argmax(predictions[0])

    confidence = np.max(predictions[0]) * 100

    # --------------------------------
    # Result
    # --------------------------------

    st.subheader("Prediction")

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # --------------------------------
    # Probabilities
    # --------------------------------

    st.subheader("Prediction Probabilities")

    probabilities = predictions[0] * 100

    for digit in range(10):

        st.write(
            f"Digit {digit}: {probabilities[digit]:.2f}%"
        )

        st.progress(
            float(predictions[0][digit])
        )