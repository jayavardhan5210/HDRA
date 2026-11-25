from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import os
import uuid

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from ecg_image_processor import process_ecg_image


app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ecg_cnn_improved_best.keras"
)


UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "HYP",
    "CD"
]


print("Loading ECG model...")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Model file not found: "
        + MODEL_PATH
    )


model = load_model(
    MODEL_PATH,
    compile=False
)


print(
    "ECG model loaded successfully."
)


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():

    return render_template(
        "hero.html"
    )


@app.route("/hero.html")
def hero():

    return render_template(
        "hero.html"
    )


@app.route("/details")
def details():

    return render_template(
        "details.html"
    )


@app.route("/details.html")
def details_html():

    return render_template(
        "details.html"
    )


@app.route("/learn")
def learn():

    return render_template(
        "learn.html"
    )


@app.route("/learn.html")
def learn_html():

    return render_template(
        "learn.html"
    )


@app.route("/result.html")
def result():

    return render_template(
        "result.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print(
        "\n========== PREDICTION REQUEST =========="
    )

    try:

        full_name = request.form.get(
            "full_name",
            "Patient"
        )

        age = float(
            request.form.get(
                "age",
                0
            )
        )

        gender = float(
            request.form.get(
                "gender",
                0
            )
        )

        bp = float(
            request.form.get(
                "bp",
                0
            )
        )

        cholesterol = float(
            request.form.get(
                "cholesterol",
                0
            )
        )

        bmi = float(
            request.form.get(
                "bmi",
                0
            )
        )

        diabetes = float(
            request.form.get(
                "diabetes",
                0
            )
        )

        smoking = float(
            request.form.get(
                "smoking",
                0
            )
        )

        family_history = float(
            request.form.get(
                "family_history",
                0
            )
        )

        exercise = float(
            request.form.get(
                "exercise",
                0
            )
        )


        ecg_file = request.files.get(
            "ecg"
        )


        if ecg_file is None:

            return jsonify({
                "status": "error",
                "message": "Please upload an ECG image."
            }), 400


        if ecg_file.filename == "":

            return jsonify({
                "status": "error",
                "message": "Please select an ECG image."
            }), 400


        if not allowed_file(
            ecg_file.filename
        ):

            return jsonify({
                "status": "error",
                "message":
                    "Only JPG, JPEG and PNG ECG images are supported."
            }), 400


        original_name = secure_filename(
            ecg_file.filename
        )


        extension = os.path.splitext(
            original_name
        )[1].lower()


        unique_name = (
            "ecg_"
            + str(uuid.uuid4())
            + extension
        )


        image_path = os.path.join(
            UPLOAD_DIR,
            unique_name
        )


        ecg_file.save(
            image_path
        )


        print(
            "ECG image saved:",
            image_path
        )


        print(
            "Processing ECG image..."
        )


        ecg_signal = process_ecg_image(
            image_path
        )


        print(
            "Processed ECG shape:",
            ecg_signal.shape
        )


        model_input = np.expand_dims(
            ecg_signal,
            axis=0
        )


        print(
            "Model input shape:",
            model_input.shape
        )


        predictions = model.predict(
            model_input,
            verbose=0
        )


        predictions = np.asarray(
            predictions
        )


        if predictions.ndim == 2:

            predictions = predictions[0]


        predictions = np.clip(
            predictions,
            0,
            1
        )


        predicted_index = int(
            np.argmax(
                predictions
            )
        )


        predicted_class = CLASSES[
            predicted_index
        ]


        confidence = float(
            predictions[
                predicted_index
            ]
        )


        print(
            "Predicted class:",
            predicted_class
        )


        print(
            "Confidence:",
            confidence
        )


        clinical_score = calculate_clinical_score(
            age,
            bp,
            cholesterol,
            bmi,
            diabetes,
            smoking,
            family_history,
            exercise
        )


        combined_score = (
            clinical_score * 0.40
            +
            confidence * 0.60
        )


        combined_score = float(
            np.clip(
                combined_score,
                0.05,
                0.98
            )
        )


        if combined_score < 0.35:

            risk_category = "Low Risk"

        elif combined_score < 0.70:

            risk_category = "Medium Risk"

        else:

            risk_category = "High Risk"


        probabilities = {}


        for i, class_name in enumerate(
            CLASSES
        ):

            probabilities[
                class_name
            ] = round(
                float(
                    predictions[i]
                ) * 100,
                2
            )


        chart_filename = (
            "risk_factors_"
            + str(uuid.uuid4())
            + ".png"
        )


        chart_path = os.path.join(
            UPLOAD_DIR,
            chart_filename
        )


        generate_risk_factor_chart(
            age,
            bp,
            cholesterol,
            bmi,
            smoking,
            family_history,
            chart_path
        )


        response = {

            "status": "success",

            "patient_name":
                full_name,

            "risk_category":
                risk_category,

            "risk_score":
                round(
                    combined_score * 100,
                    2
                ),

            "ecg_prediction":
                predicted_class,

            "ecg_confidence":
                round(
                    confidence * 100,
                    2
                ),

            "probabilities":
                probabilities,

            "shap_image_url":
                "/static/uploads/"
                + chart_filename

        }


        print(
            "Prediction completed successfully."
        )


        print(
            "=======================================\n"
        )


        return jsonify(
            response
        )


    except ValueError as e:

        print(
            "Processing error:",
            str(e)
        )


        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


    except Exception as e:

        print(
            "Unexpected error:",
            str(e)
        )


        return jsonify({
            "status": "error",
            "message":
                "Prediction failed: "
                + str(e)
        }), 500


def calculate_clinical_score(
    age,
    bp,
    cholesterol,
    bmi,
    diabetes,
    smoking,
    family_history,
    exercise
):

    age_score = min(
        max(age / 100.0, 0),
        1
    )


    bp_score = min(
        max(bp / 200.0, 0),
        1
    )


    cholesterol_score = min(
        max(cholesterol / 300.0, 0),
        1
    )


    bmi_score = min(
        max(bmi / 40.0, 0),
        1
    )


    smoking_score = (
        1.0
        if smoking > 0
        else 0.0
    )


    diabetes_score = (
        1.0
        if diabetes > 0
        else 0.0
    )


    family_score = (
        1.0
        if family_history > 0
        else 0.0
    )


    if exercise == 0:

        exercise_score = 1.0

    elif exercise == 1:

        exercise_score = 0.5

    else:

        exercise_score = 0.1


    score = (

        age_score * 0.15

        +

        bp_score * 0.20

        +

        cholesterol_score * 0.20

        +

        bmi_score * 0.10

        +

        diabetes_score * 0.15

        +

        smoking_score * 0.10

        +

        family_score * 0.05

        +

        exercise_score * 0.05

    )


    return float(
        np.clip(
            score,
            0,
            1
        )
    )


def generate_risk_factor_chart(
    age,
    bp,
    cholesterol,
    bmi,
    smoking,
    family_history,
    save_path
):

    features = [
        "Blood Pressure",
        "Cholesterol",
        "Age",
        "BMI",
        "Smoking",
        "Family History"
    ]


    values = [
        min(bp / 180.0, 1.0),
        min(cholesterol / 300.0, 1.0),
        min(age / 100.0, 1.0),
        min(bmi / 40.0, 1.0),
        1.0 if smoking else 0.0,
        1.0 if family_history else 0.0
    ]


    plt.figure(
        figsize=(8, 5)
    )


    plt.barh(
        features,
        values
    )


    plt.xlabel(
        "Relative Factor Value"
    )


    plt.title(
        "Patient Heart Risk Factors"
    )


    plt.xlim(
        0,
        1
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=150
    )


    plt.close()


if __name__ == "__main__":

    print(
        "\nStarting Heart Disease Risk Assessment Agent..."
    )


    app.run(
        debug=True,
        port=5000
    )