import streamlit as st
from pathlib import Path
import json
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Risk Prediction",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_heart_model():

    model = joblib.load(
        MODEL_DIR / "heart_risk_model.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "heart_risk_scaler.pkl"
    )

    with open(
        MODEL_DIR / "heart_features.json",
        "r",
        encoding="utf-8"
    ) as f:

        features = json.load(f)

    return model, scaler, features


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    "heart_risk_model.pkl",
    "heart_risk_scaler.pkl",
    "heart_features.json"
]

missing_files = [
    file for file in required_files
    if not (MODEL_DIR / file).exists()
]


if missing_files:

    st.error("❌ Heart model files are missing.")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Copy the trained heart model files from Colab "
        "into the models folder."
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    model, scaler, features = load_heart_model()

except Exception as e:

    st.error(
        f"❌ Error loading heart model: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("❤️ Heart Disease Risk Prediction")

st.write(
    """
    Enter the patient information below. The trained machine
    learning model will estimate the predicted class and
    positive-class probability.
    """
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with col2:

    st.metric(
        "Features",
        len(features)
    )

with col3:

    st.metric(
        "Preprocessing",
        "StandardScaler"
    )


st.divider()


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("📝 Patient Information")

st.caption(
    "The inputs below are generated from the exact feature list "
    "saved during model training."
)


values = {}


# ============================================================
# FEATURE INPUTS
# ============================================================

cols = st.columns(2)


for index, feature in enumerate(features):

    feature_name = str(feature).strip()

    label = (
        feature_name
        .replace("_", " ")
        .title()
    )


    with cols[index % 2]:


        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        if feature_name == "age":

            values[feature_name] = st.number_input(
                label,
                min_value=1.0,
                max_value=120.0,
                value=50.0,
                step=1.0
            )


        # ----------------------------------------------------
        # SEX
        # ----------------------------------------------------

        elif feature_name == "sex":

            values[feature_name] = st.selectbox(
                label,
                options=[0, 1],
                format_func=lambda x:
                    "Female (0)" if x == 0
                    else "Male (1)"
            )


        # ----------------------------------------------------
        # CHEST PAIN
        # ----------------------------------------------------

        elif feature_name == "chest_pain_type":

            values[feature_name] = st.number_input(
                label,
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=1.0
            )


        # ----------------------------------------------------
        # RESTING BLOOD PRESSURE
        # ----------------------------------------------------

        elif feature_name == "resting_bp_s":

            values[feature_name] = st.number_input(
                label,
                min_value=50.0,
                max_value=250.0,
                value=120.0,
                step=1.0
            )


        # ----------------------------------------------------
        # CHOLESTEROL
        # ----------------------------------------------------

        elif feature_name == "cholesterol":

            values[feature_name] = st.number_input(
                label,
                min_value=50.0,
                max_value=700.0,
                value=200.0,
                step=1.0
            )


        # ----------------------------------------------------
        # RESTING ECG
        # ----------------------------------------------------

        elif feature_name == "resting_ecg":

            values[feature_name] = st.number_input(
                label,
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=1.0
            )


        # ----------------------------------------------------
        # MAX HEART RATE
        # ----------------------------------------------------

        elif feature_name == "max_heart_rate":

            values[feature_name] = st.number_input(
                label,
                min_value=50.0,
                max_value=250.0,
                value=150.0,
                step=1.0
            )


        # ----------------------------------------------------
        # EXERCISE ANGINA
        # ----------------------------------------------------

        elif feature_name == "exercise_angina":

            values[feature_name] = st.selectbox(
                label,
                options=[0, 1],
                format_func=lambda x:
                    "No (0)" if x == 0
                    else "Yes (1)"
            )


        # ----------------------------------------------------
        # OLDPEAK
        # ----------------------------------------------------

        elif feature_name == "oldpeak":

            values[feature_name] = st.number_input(
                label,
                min_value=-5.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )


        # ----------------------------------------------------
        # ST SLOPE
        # ----------------------------------------------------

        elif feature_name == "ST_slope":

            values[feature_name] = st.number_input(
                label,
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=1.0
            )


        # ----------------------------------------------------
        # OTHER FEATURES
        # ----------------------------------------------------

        else:

            values[feature_name] = st.number_input(
                label,
                value=0.0
            )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "❤️ Calculate Heart Risk",
    type="primary",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = {
            feature: values[feature]
            for feature in features
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )


        # ----------------------------------------------------
        # APPLY SAME SCALER USED DURING TRAINING
        # ----------------------------------------------------

        scaled_data = scaler.transform(
            input_df
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            scaled_data
        )[0]


        probabilities = model.predict_proba(
            scaled_data
        )[0]


        # ----------------------------------------------------
        # FIND POSITIVE CLASS
        # ----------------------------------------------------

        classes = list(
            model.classes_
        )


        if 1 in classes:

            positive_index = classes.index(1)

        else:

            positive_index = len(
                classes
            ) - 1


        positive_probability = probabilities[
            positive_index
        ]


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "Prediction completed successfully!"
        )

        st.subheader(
            "❤️ Heart Risk Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Predicted Class",
                str(prediction)
            )


        with col2:

            st.metric(
                "Positive-Class Probability",
                f"{positive_probability * 100:.2f}%"
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.write(
            "Model probability"
        )

        st.progress(
            float(positive_probability)
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        if prediction == 1:

            st.error(
                "The model classified this input "
                "as the positive class."
            )

        else:

            st.success(
                "The model classified this input "
                "as the negative class."
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Input Summary"
        )

        summary_df = pd.DataFrame(
            {
                "Feature": features,
                "Value": [
                    values[feature]
                    for feature in features
                ]
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # ALL CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Class Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    str(c)
                    for c in classes
                ],
                "Probability": [
                    float(p) * 100
                    for p in probabilities
                ]
            }
        )

        probability_df["Probability"] = (
            probability_df["Probability"]
            .round(2)
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application is an educational/research prototype.

    The prediction is generated by a machine learning model
    and should NOT be interpreted as a medical diagnosis or
    clinical risk assessment.

    Please consult a qualified healthcare professional for
    medical evaluation and treatment decisions.
    """
)