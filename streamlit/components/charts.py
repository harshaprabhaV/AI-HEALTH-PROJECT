import streamlit as st
import pandas as pd


# ============================================================
# DISEASE PROBABILITY CHART
# ============================================================

def disease_probability_chart(results):
    """
    Display disease prediction probabilities.

    Expected format:
    [
        {
            "disease": "Disease Name",
            "probability": 0.85
        }
    ]
    """

    if not results:
        st.info("No prediction results available.")
        return

    chart_data = pd.DataFrame(
        {
            "Disease": [
                result["disease"]
                for result in results
            ],
            "Probability (%)": [
                result["probability"] * 100
                for result in results
            ]
        }
    )

    chart_data = chart_data.set_index(
        "Disease"
    )

    st.bar_chart(
        chart_data
    )


# ============================================================
# CLASS PROBABILITY CHART
# ============================================================

def class_probability_chart(
    class_probabilities
):
    """
    Display classification probabilities.

    Example:
    {
        "0": 0.25,
        "1": 0.75
    }
    """

    if not class_probabilities:
        st.info("No probability data available.")
        return

    chart_data = pd.DataFrame(
        {
            "Class": list(
                class_probabilities.keys()
            ),
            "Probability (%)": [
                value * 100
                for value
                in class_probabilities.values()
            ]
        }
    )

    chart_data = chart_data.set_index(
        "Class"
    )

    st.bar_chart(
        chart_data
    )


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

def feature_importance_chart(
    importance_items,
    top_n=10
):
    """
    Display top model feature importances.

    Expected format:
    [
        ("symptom_name", 0.12),
        ("another_symptom", 0.08)
    ]
    """

    if not importance_items:
        st.info(
            "No feature importance data available."
        )
        return


    top_items = importance_items[:top_n]


    chart_data = pd.DataFrame(
        top_items,
        columns=[
            "Feature",
            "Importance"
        ]
    )


    chart_data["Feature"] = (
        chart_data["Feature"]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False
        )
        .str.title()
    )


    chart_data = chart_data.set_index(
        "Feature"
    )


    st.bar_chart(
        chart_data
    )


# ============================================================
# SIMPLE METRIC CHART
# ============================================================

def metric_bar_chart(
    labels,
    values,
    title="Chart"
):
    """
    Display a simple bar chart.
    """

    if not labels or not values:

        st.info(
            "No chart data available."
        )

        return


    chart_data = pd.DataFrame(
        {
            "Category": labels,
            "Value": values
        }
    )


    chart_data = chart_data.set_index(
        "Category"
    )


    st.subheader(
        title
    )


    st.bar_chart(
        chart_data
    )