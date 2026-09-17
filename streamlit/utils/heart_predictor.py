from pathlib import Path
import json
import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# LOAD HEART MODEL
# ============================================================

def load_heart_model():

    model_path = MODEL_DIR / "heart_risk_model.pkl"

    scaler_path = MODEL_DIR / "heart_risk_scaler.pkl"

    features_path = MODEL_DIR / "heart_features.json"


    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    required_files = [
        model_path,
        scaler_path,
        features_path
    ]


    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------

    scaler = joblib.load(
        scaler_path
    )


    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    with open(
        features_path,
        "r",
        encoding="utf-8"
    ) as file:

        features = json.load(
            file
        )


    return (
        model,
        scaler,
        features
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_heart_input(
    values,
    features
):

    # Create dictionary in EXACT
    # training feature order

    input_data = {}

    for feature in features:

        if feature not in values:

            raise ValueError(
                f"Missing feature: {feature}"
            )

        input_data[feature] = values[
            feature
        ]


    # Convert to DataFrame

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )


    return input_df


# ============================================================
# PREDICT HEART RISK
# ============================================================

def predict_heart_risk(
    values
):

    (
        model,
        scaler,
        features
    ) = load_heart_model()


    # --------------------------------------------------------
    # PREPARE INPUT
    # --------------------------------------------------------

    input_df = prepare_heart_input(
        values,
        features
    )


    # --------------------------------------------------------
    # APPLY TRAINING SCALER
    # --------------------------------------------------------

    scaled_input = scaler.transform(
        input_df
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        scaled_input
    )[0]


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        scaled_input
    )[0]


    # --------------------------------------------------------
    # MODEL CLASSES
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )


    # --------------------------------------------------------
    # POSITIVE CLASS
    # --------------------------------------------------------

    if 1 in classes:

        positive_index = classes.index(
            1
        )

    else:

        positive_index = len(
            classes
        ) - 1


    positive_probability = probabilities[
        positive_index
    ]


    # --------------------------------------------------------
    # ALL PROBABILITIES
    # --------------------------------------------------------

    class_probabilities = {}

    for index, class_value in enumerate(
        classes
    ):

        class_probabilities[
            str(class_value)
        ] = float(
            probabilities[index]
        )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "prediction": int(prediction),

        "positive_probability": float(
            positive_probability
        ),

        "class_probabilities":
            class_probabilities,

        "features":
            features
    }


# ============================================================
# GET HEART FEATURES
# ============================================================

def get_heart_features():

    (
        _,
        _,
        features
    ) = load_heart_model()


    return features