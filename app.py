
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open("Heart_prediction.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaled_model.pkl", "rb") as file:
    scaler = pickle.load(file)


# Exact feature order used during training
FEATURES = [
    "age_yeo_trim",
    "sex_yeo_trim",
    "cp_yeo_trim",
    "thalach_yeo_trim",
    "oldpeak_yeo_trim",
    "slope_yeo_trim",
    "thal_yeo_trim"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from HTML
        age = float(request.form["age"])
        sex_value = request.form["sex"]
        cp = float(request.form["cp"])
        thalach = float(request.form["thalach"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        thal = float(request.form["thal"])


        # Male = 1
        # Female = 0
        if sex_value == "Male":
            sex = 1
        else:
            sex = 0


        # Create input dataframe
        input_data = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                thalach,
                oldpeak,
                slope,
                thal
            ]],
            columns=FEATURES
        )


        # Scale input
        scaled_data = scaler.transform(input_data)


        # Prediction
        prediction = model.predict(scaled_data)[0]


        # Result
        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"


        # Probability
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(scaled_data)[0]

            probability = round(
                max(probabilities) * 100,
                2
            )


        return render_template(
            "index.html",
            prediction=result,
            probability=probability
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error: " + str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
