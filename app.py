"""
Breast Cancer Prediction Web App
================================
Production-ready Flask application with lazy model loading to avoid timeout.
"""

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# App Configuration
# ---------------------------------------------------------
app = Flask(__name__)

# Logging configuration (Render-compatible)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "model/breast_cancer_model.h5"
THRESHOLD = 0.5

# ---------------------------------------------------------
# Load dataset & scaler (lightweight)
# ---------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
FEATURE_NAMES = list(X.columns)

scaler = StandardScaler()
scaler.fit(X)

# ---------------------------------------------------------
# Lazy-load model (avoids Render startup timeout)
# ---------------------------------------------------------
model = None


def get_model():
    global model
    if model is None:
        logger.info("Loading ANN model...")
        from tensorflow.keras.models import load_model  # import here
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    return model


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    prediction_class = None

    if request.method == "POST":
        try:
            input_features = [
                float(request.form[feature])
                for feature in FEATURE_NAMES
            ]
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Lazy-load model
            model_instance = get_model()
            prob = float(model_instance.predict(input_scaled, verbose=0)[0][0])
            probability = round(prob, 4)

            if prob >= THRESHOLD:
                result = "Benign Tumor"
                prediction_class = "benign"
            else:
                result = "Malignant Tumor"
                prediction_class = "malignant"

            logger.info(f"Prediction made | Result: {result} | Probability: {probability}")

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            result = "Invalid input values. Please check your entries."
            prediction_class = "error"

    return render_template(
        "index.html",
        features=FEATURE_NAMES,
        prediction=result,
        prediction_class=prediction_class,
        probability=probability
    )


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        status="ok",
        model_loaded=model is not None
    ), 200


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
