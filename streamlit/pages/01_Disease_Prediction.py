import streamlit as st
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_disease_model():

    model = joblib.load(
        MODEL_DIR / "disease_model.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "disease_label_encoder.pkl"
    )

    with open(
        MODEL_DIR / "symptom_columns.json",
        "r",
        encoding="utf-8"
    ) as f:
        symptom_columns = json.load(f)

    with open(
        MODEL_DIR / "symptom_importances.json",
        "r",
        encoding="utf-8"
    ) as f:
        symptom_importances = json.load(f)

    return (
        model,
        label_encoder,
        symptom_columns,
        symptom_importances
    )


# ============================================================
# LOAD SUPPORTING DATA
# ============================================================

@st.cache_data
def load_description_data():

    file_path = DATA_DIR / "symptom_description.csv"

    if not file_path.exists():
        return None

    for encoding in ["utf-8", "utf-8-sig", "latin1"]:

        try:
            return pd.read_csv(
                file_path,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    return None


@st.cache_data
def load_precaution_data():

    file_path = DATA_DIR / "symptom_precaution.csv"

    if not file_path.exists():
        return None

    for encoding in ["utf-8", "utf-8-sig", "latin1"]:

        try:
            return pd.read_csv(
                file_path,
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    return None


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Disease Prediction")

st.write(
    """
    Select the symptoms you are experiencing and the trained
    machine learning model will return the top possible disease
    predictions.
    """
)

st.divider()


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    "disease_model.pkl",
    "disease_label_encoder.pkl",
    "symptom_columns.json",
    "symptom_importances.json"
]

missing_files = []

for file in required_files:

    if not (MODEL_DIR / file).exists():
        missing_files.append(file)


if missing_files:

    st.error("❌ Required model files are missing:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Copy the trained files from your Colab output into the models folder."
    )

    st.stop()


# ============================================================
# LOAD ARTIFACTS
# ============================================================

try:

    (
        model,
        label_encoder,
        symptom_columns,
        symptom_importances
    ) = load_disease_model()

except Exception as e:

    st.error(
        f"❌ Error loading disease model: {e}"
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Disease Classes",
        len(label_encoder.classes_)
    )

with col2:

    st.metric(
        "Symptoms",
        len(symptom_columns)
    )

with col3:

    st.metric(
        "Model",
        "Random Forest"
    )


st.divider()


# ============================================================
# SYMPTOM SELECTION
# ============================================================

st.subheader("🔍 Select Your Symptoms")

st.write(
    "Choose one or more symptoms from the list below."
)


# Convert names for display

display_symptoms = {
    symptom: str(symptom).replace("_", " ").title()
    for symptom in symptom_columns
}


selected_display = st.multiselect(
    "Symptoms",
    options=list(display_symptoms.values()),
    placeholder="Search and select symptoms..."
)


# Convert selected display names back to original column names

reverse_symptoms = {
    value: key
    for key, value in display_symptoms.items()
}

selected_symptoms = [
    reverse_symptoms[s]
    for s in selected_display
]


# ============================================================
# SELECTED SYMPTOMS DISPLAY
# ============================================================

if selected_symptoms:

    st.subheader("✅ Selected Symptoms")

    selected_cols = st.columns(
        min(len(selected_symptoms), 4)
    )

    for i, symptom in enumerate(selected_symptoms):

        with selected_cols[i % len(selected_cols)]:

            st.info(
                str(symptom)
                .replace("_", " ")
                .title()
            )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(symptoms):

    # Create empty feature vector

    input_data = np.zeros(
        len(symptom_columns)
    )


    # Activate selected symptoms

    for symptom in symptoms:

        if symptom in symptom_columns:

            index = symptom_columns.index(symptom)

            input_data[index] = 1


    # Convert to DataFrame

    input_df = pd.DataFrame(
        [input_data],
        columns=symptom_columns
    )


    # Prediction probabilities

    probabilities = model.predict_proba(
        input_df
    )[0]


    # Get top 3 predictions

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]


    results = []

    for index in top_indices:

        disease = label_encoder.inverse_transform(
            [index]
        )[0]

        probability = probabilities[index]

        results.append(
            {
                "disease": disease,
                "probability": probability
            }
        )


    return results


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Disease",
    type="primary",
    use_container_width=True
):

    if len(selected_symptoms) == 0:

        st.warning(
            "⚠️ Please select at least one symptom."
        )

    else:

        try:

            results = predict_disease(
                selected_symptoms
            )


            # ==================================================
            # RESULTS
            # ==================================================

            st.success(
                "Prediction completed successfully!"
            )

            st.subheader(
                "🧠 Top 3 Model Predictions"
            )


            for i, result in enumerate(
                results,
                start=1
            ):

                disease = result["disease"]

                probability = result["probability"]


                st.markdown(
                    f"### {i}. {disease}"
                )


                st.progress(
                    float(probability)
                )


                st.write(
                    f"Model probability: "
                    f"**{probability * 100:.2f}%**"
                )


                st.divider()


            # ==================================================
            # TOP PREDICTION INFORMATION
            # ==================================================

            top_disease = results[0]["disease"]


            st.subheader(
                "📋 Disease Information"
            )


            description_df = load_description_data()


            if description_df is not None:

                columns_lower = {
                    str(c).lower(): c
                    for c in description_df.columns
                }


                disease_column = None
                description_column = None


                # Find disease column

                for name in [
                    "disease",
                    "prognosis"
                ]:

                    if name in columns_lower:

                        disease_column = columns_lower[name]

                        break


                # Find description column

                for name in [
                    "description",
                    "desc"
                ]:

                    if name in columns_lower:

                        description_column = columns_lower[name]

                        break


                if (
                    disease_column is not None
                    and description_column is not None
                ):

                    matches = description_df[
                        description_df[disease_column]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        str(top_disease)
                        .strip()
                        .lower()
                    ]


                    if not matches.empty:

                        st.write(
                            matches.iloc[0][
                                description_column
                            ]
                        )

                    else:

                        st.info(
                            "No description found for this prediction."
                        )

                else:

                    st.info(
                        "Description columns were not detected."
                    )

            else:

                st.info(
                    "symptom_description.csv was not found."
                )


            # ==================================================
            # PRECAUTIONS
            # ==================================================

            precaution_df = load_precaution_data()


            if precaution_df is not None:

                st.subheader(
                    "🛡️ Available Precautions"
                )


                columns_lower = {
                    str(c).lower(): c
                    for c in precaution_df.columns
                }


                disease_column = None


                for name in [
                    "disease",
                    "prognosis"
                ]:

                    if name in columns_lower:

                        disease_column = columns_lower[name]

                        break


                if disease_column is not None:

                    matches = precaution_df[
                        precaution_df[disease_column]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        str(top_disease)
                        .strip()
                        .lower()
                    ]


                    if not matches.empty:

                        precaution_row = matches.iloc[0]


                        for column in precaution_df.columns:

                            if column == disease_column:
                                continue


                            value = precaution_row[column]


                            if (
                                pd.notna(value)
                                and str(value).strip()
                            ):

                                st.write(
                                    f"• {value}"
                                )


            # ==================================================
            # MODEL TRANSPARENCY
            # ==================================================

            st.subheader(
                "📊 Model Feature Importance"
            )

            st.caption(
                "These are global Random Forest feature importances, "
                "not a clinical explanation for this individual prediction."
            )


            # Handle dictionary or list format

            if isinstance(
                symptom_importances,
                dict
            ):

                importance_items = list(
                    symptom_importances.items()
                )

            else:

                importance_items = list(
                    zip(
                        symptom_columns,
                        symptom_importances
                    )
                )


            importance_items = sorted(
                importance_items,
                key=lambda x: float(x[1]),
                reverse=True
            )[:10]


            importance_df = pd.DataFrame(
                importance_items,
                columns=[
                    "Symptom",
                    "Importance"
                ]
            )


            importance_df["Symptom"] = (
                importance_df["Symptom"]
                .astype(str)
                .str.replace("_", " ")
                .str.title()
            )


            st.bar_chart(
                importance_df.set_index(
                    "Symptom"
                )
            )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application provides machine-learning predictions for
    educational and research purposes only.

    The predicted disease is NOT a medical diagnosis.
    Do not use this result to start, stop, or change treatment.

    Please consult a qualified healthcare professional for
    diagnosis and medical advice.
    """
)