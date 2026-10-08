import streamlit as st
import keras
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path


# -----------------------------------
# Load Model
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_digit_model.keras"

model = keras.models.load_model(MODEL_PATH)


# -----------------------------------
# Streamlit Page
# -----------------------------------

st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢"
)

st.title("🔢 MNIST Digit Predictor")

st.write("Upload a handwritten digit image.")


# -----------------------------------
# Upload Image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # -----------------------------------
    # Open Image
    # -----------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("L")

    st.subheader("Uploaded Image")

    st.image(
        original_image,
        width=200
    )


    # -----------------------------------
    # Convert Image
    # -----------------------------------

    image = original_image

    # Resize
    image = image.resize((28, 28))

    image_array = np.array(image)


    # -----------------------------------
    # Check Background
    # -----------------------------------

    # If image has white background,
    # invert it to match MNIST style.

    if image_array.mean() > 127:

        image_array = 255 - image_array


    # -----------------------------------
    # Normalize
    # -----------------------------------

    image_array = (
        image_array.astype("float32") / 255.0
    )


    # -----------------------------------
    # Show Processed Image
    # -----------------------------------

    st.subheader("Image Given to Model")

    st.image(
        image_array,
        width=200
    )


    # -----------------------------------
    # Add Batch Dimension
    # -----------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------
    # Prediction
    # -----------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )


    predicted_digit = np.argmax(
        predictions[0]
    )

    confidence = (
        np.max(predictions[0]) * 100
    )


    # -----------------------------------
    # Result
    # -----------------------------------

    st.subheader("Prediction")

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # -----------------------------------
    # All Probabilities
    # -----------------------------------

    st.subheader("Prediction Probabilities")

    probabilities = predictions[0] * 100

    for digit in range(10):

        st.write(
            f"Digit {digit}: "
            f"{probabilities[digit]:.2f}%"
        )

        st.progress(
            float(predictions[0][digit])
        )