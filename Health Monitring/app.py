from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model parameters
with open("health_model.pkl", "rb") as f:
    model_params = pickle.load(f)

coefficients = np.array(model_params["coefficients"])
intercept = model_params["intercept"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = {}

    if request.method == "POST":
        try:
            # Get user inputs
            input_values = {
                "bp_systolic": float(request.form["bp_systolic"]),
                "bp_diastolic": float(request.form["bp_diastolic"]),
                "sugar_level": float(request.form["sugar_level"]),
                "cholesterol": float(request.form["cholesterol"]),
                "haemoglobin": float(request.form["haemoglobin"]),
            }

            # Convert to array and predict
            input_array = np.array(list(input_values.values())).reshape(1, -1)
            prediction = np.dot(input_array, coefficients) + intercept
            prediction = round(prediction[0], 2)

        except ValueError:
            prediction = "Invalid Input"

    return render_template("index.html", prediction=prediction, input_values=input_values)

if __name__ == "__main__":
    app.run(debug=True)
