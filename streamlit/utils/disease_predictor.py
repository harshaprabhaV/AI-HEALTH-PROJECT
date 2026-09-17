from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

def load_disease_model():

    model_path = MODEL_DIR / "disease_model.pkl"

    encoder_path = MODEL_DIR / "disease_label_encoder.pkl"

    symptoms_path = MODEL_DIR / "symptom_columns.json"

    importance_path = MODEL_DIR / "symptom_importances.json"


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        model_path,
        encoder_path,
        symptoms_path,
        importance_path
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
    # LOAD LABEL ENCODER
    # --------------------------------------------------------

    label_encoder = joblib.load(
        encoder_path
    )


    # --------------------------------------------------------
    # LOAD SYMPTOM COLUMNS
    # --------------------------------------------------------

    with open(
        symptoms_path,
        "r",
        encoding="utf-8"
    ) as file:

        symptom_columns = json.load(
            file
        )


    # --------------------------------------------------------
    # LOAD FEATURE IMPORTANCE
    # --------------------------------------------------------

    with open(
        importance_path,
        "r",
        encoding="utf-8"
    ) as file:

        symptom_importances = json.load(
            file
        )


    return (
        model,
        label_encoder,
        symptom_columns,
        symptom_importances
    )


# ============================================================
# CREATE INPUT VECTOR
# ============================================================

def create_symptom_vector(
    selected_symptoms,
    symptom_columns
):

    # Create all-zero feature vector

    vector = np.zeros(
        len(symptom_columns),
        dtype=float
    )


    # Convert selected symptoms to lowercase
    # for safer matching

    selected_set = {
        str(symptom).strip().lower()
        for symptom in selected_symptoms
    }


    # Activate selected symptoms

    for index, symptom in enumerate(
        symptom_columns
    ):

        if (
            str(symptom)
            .strip()
            .lower()
            in selected_set
        ):

            vector[index] = 1.0


    return vector


# ============================================================
# PREDICT DISEASE
# ============================================================

def predict_disease(
    selected_symptoms,
    top_k=3
):

    (
        model,
        label_encoder,
        symptom_columns,
        symptom_importances
    ) = load_disease_model()


    # --------------------------------------------------------
    # CREATE FEATURE VECTOR
    # --------------------------------------------------------

    input_vector = create_symptom_vector(
        selected_symptoms,
        symptom_columns
    )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_vector],
        columns=symptom_columns
    )


    # --------------------------------------------------------
    # GET PROBABILITIES
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_df
    )[0]


    # --------------------------------------------------------
    # TOP K PREDICTIONS
    # --------------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]


    results = []


    for index in top_indices:

        disease = label_encoder.inverse_transform(
            [index]
        )[0]


        probability = probabilities[
            index
        ]


        results.append(
            {
                "disease": str(disease),
                "probability": float(
                    probability
                )
            }
        )


    return results


# ============================================================
# GET MODEL FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    (
        model,
        label_encoder,
        symptom_columns,
        symptom_importances
    ) = load_disease_model()


    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(
        symptom_importances,
        dict
    ):

        importance_items = [
            (
                symptom,
                float(value)
            )

            for symptom, value
            in symptom_importances.items()
        ]


    # --------------------------------------------------------
    # List format
    # --------------------------------------------------------

    elif isinstance(
        symptom_importances,
        list
    ):

        importance_items = list(
            zip(
                symptom_columns,
                [
                    float(value)
                    for value
                    in symptom_importances
                ]
            )
        )


    else:

        # Fallback to Random Forest importance

        importance_items = list(
            zip(
                symptom_columns,
                model.feature_importances_
            )
        )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    importance_items.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return importance_items


# ============================================================
# GET AVAILABLE SYMPTOMS
# ============================================================

def get_symptom_columns():

    (
        _,
        _,
        symptom_columns,
        _
    ) = load_disease_model()


    return symptom_columns