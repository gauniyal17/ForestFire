from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Ridge regression model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extracting only the required features from form input
            features = [
                float(request.form.get("Temperature")),
                float(request.form.get("RH")),
                float(request.form.get("Ws")),
                float(request.form.get("Rain")),
                float(request.form.get("FFMC")),
                float(request.form.get("DMC")),
                float(request.form.get("ISI")),
                float(request.form.get("Classes")),
                float(request.form.get("Region"))
            ]

            # Convert to NumPy array and reshape
            input_data = np.array(features).reshape(1, -1)

            # Scale the input data
            scaled_data = standard_scaler.transform(input_data)

            # Make prediction
            prediction = ridge_model.predict(scaled_data)[0]

            return jsonify({"prediction": round(prediction, 2)})

        except Exception as e:
            return jsonify({"error": str(e)})

    else:
        return render_template("home.html")  # Rendering home.html for GET requests

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
