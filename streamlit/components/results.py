import streamlit as st


# ============================================================
# DISEASE PREDICTION RESULTS
# ============================================================

def show_disease_results(results):
    """
    Display top disease predictions.

    Expected format:
    [
        {
            "disease": "Disease Name",
            "probability": 0.85
        }
    ]
    """

    if not results:

        st.info(
            "No prediction results available."
        )

        return


    st.subheader(
        "🧠 Prediction Results"
    )


    for index, result in enumerate(
        results,
        start=1
    ):

        disease = result.get(
            "disease",
            "Unknown"
        )

        probability = float(
            result.get(
                "probability",
                0
            )
        )


        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. {disease}"
            )


            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )


            st.write(
                f"Model probability: "
                f"**{probability * 100:.2f}%**"
            )


# ============================================================
# HEART RISK RESULT
# ============================================================

def show_heart_result(
    prediction,
    probability
):
    """
    Display heart-risk model result.
    """

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
            f"{probability * 100:.2f}%"
        )


    st.write(
        "Model probability"
    )


    st.progress(
        min(
            max(
                float(probability),
                0.0
            ),
            1.0
        )
    )


# ============================================================
# RAG RESULTS
# ============================================================

def show_rag_results(
    results
):
    """
    Display medical knowledge retrieval results.

    Expected format:
    [
        {
            "question": "...",
            "answer": "...",
            "score": 0.75
        }
    ]
    """

    if not results:

        st.info(
            "No relevant medical information found."
        )

        return


    st.subheader(
        "📚 Retrieved Medical Information"
    )


    for index, result in enumerate(
        results,
        start=1
    ):

        question = result.get(
            "question",
            "Unknown question"
        )

        answer = result.get(
            "answer",
            "No answer available."
        )

        score = float(
            result.get(
                "score",
                0
            )
        )


        with st.container(
            border=True
        ):

            st.markdown(
                f"### {index}. {question}"
            )


            st.caption(
                f"Similarity Score: {score:.4f}"
            )


            st.markdown(
                "**Answer:**"
            )


            st.write(
                answer
            )


# ============================================================
# GENERAL SUCCESS RESULT
# ============================================================

def show_success_result(
    title,
    message
):

    st.success(
        f"**{title}**\n\n{message}"
    )


# ============================================================
# GENERAL ERROR RESULT
# ============================================================

def show_error_result(
    title,
    message
):

    st.error(
        f"**{title}**\n\n{message}"
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

def show_medical_disclaimer():

    st.warning(
        """
        ⚠️ **Medical Disclaimer**

        This application is intended for educational and
        research purposes only.

        Model predictions and retrieved information are not
        medical diagnoses or personalized medical advice.

        Consult a qualified healthcare professional for
        medical evaluation and treatment decisions.
        """
    )