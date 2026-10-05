from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/churn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    data = {
        "gender": request.form["gender"],
        "SeniorCitizen": int(request.form["SeniorCitizen"]),
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": int(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }

    # Convert dictionary into DataFrame
    input_data = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is not likely to churn"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)