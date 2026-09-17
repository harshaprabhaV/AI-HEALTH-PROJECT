import streamlit as st
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
    page_title="Medical Assistant",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Medical Knowledge Assistant")

st.write(
    """
    Ask a medical question and the system will retrieve the
    most relevant information from the MedQuAD medical
    question-answer knowledge base.
    """
)

st.caption(
    "Retrieval method: TF-IDF + Cosine Similarity"
)

st.divider()


# ============================================================
# REQUIRED FILES
# ============================================================

required_model_files = [
    "rag_tfidf_vectorizer.pkl",
    "rag_tfidf_matrix.pkl"
]

missing_files = []

for file in required_model_files:

    if not (MODEL_DIR / file).exists():
        missing_files.append(file)


if not (DATA_DIR / "medquad_qa.csv").exists():

    missing_files.append(
        "medquad_qa.csv"
    )


if missing_files:

    st.error(
        "❌ Required RAG files are missing."
    )

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Copy the RAG model files from Colab into models/ "
        "and medquad_qa.csv into data/."
    )

    st.stop()


# ============================================================
# LOAD RAG ARTIFACTS
# ============================================================

@st.cache_resource
def load_rag_model():

    vectorizer = joblib.load(
        MODEL_DIR / "rag_tfidf_vectorizer.pkl"
    )

    tfidf_matrix = joblib.load(
        MODEL_DIR / "rag_tfidf_matrix.pkl"
    )

    return vectorizer, tfidf_matrix


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_medical_dataset():

    file_path = DATA_DIR / "medquad_qa.csv"

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding
            )

            return df

        except UnicodeDecodeError:

            continue

    raise ValueError(
        "Could not decode medquad_qa.csv"
    )


# ============================================================
# LOAD
# ============================================================

try:

    vectorizer, tfidf_matrix = load_rag_model()

    qa_df = load_medical_dataset()

except Exception as e:

    st.error(
        f"❌ Error loading Medical Assistant: {e}"
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

required_columns = [
    "question",
    "answer"
]

missing_columns = [
    column
    for column in required_columns
    if column not in qa_df.columns
]


if missing_columns:

    st.error(
        f"❌ Missing columns in medquad_qa.csv: "
        f"{missing_columns}"
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(qa_df.columns)
    )

    st.stop()


qa_df = qa_df.dropna(
    subset=[
        "question",
        "answer"
    ]
).reset_index(drop=True)


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Knowledge Base",
        "MedQuAD"
    )

with col2:

    st.metric(
        "Q&A Records",
        f"{len(qa_df):,}"
    )

with col3:

    st.metric(
        "Retrieval Method",
        "TF-IDF"
    )


st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "💬 Ask Your Question"
)

question = st.text_area(
    "Medical Question",
    placeholder=(
        "Example: What are the symptoms of diabetes?"
    ),
    height=130
)


# ============================================================
# TOP-K SELECTION
# ============================================================

top_k = st.slider(
    "Number of answers to retrieve",
    min_value=1,
    max_value=5,
    value=3
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def retrieve_answers(
    user_question,
    number_of_results=3
):

    # Convert user question into TF-IDF vector

    query_vector = vectorizer.transform(
        [user_question]
    )


    # Calculate cosine similarity

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()


    # Get highest similarity indices

    top_indices = similarities.argsort()[
        ::-1
    ][:number_of_results]


    results = []


    for index in top_indices:

        # Make sure index exists in dataset

        if index >= len(qa_df):

            continue


        results.append(
            {
                "question": str(
                    qa_df.iloc[index]["question"]
                ),

                "answer": str(
                    qa_df.iloc[index]["answer"]
                ),

                "score": float(
                    similarities[index]
                )
            }
        )


    return results


# ============================================================
# SEARCH BUTTON
# ============================================================

if st.button(
    "🔎 Search Medical Knowledge",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "⚠️ Please enter a medical question."
        )

    else:

        try:

            results = retrieve_answers(
                question.strip(),
                top_k
            )


            # =================================================
            # RESULTS
            # =================================================

            st.success(
                "Medical knowledge retrieved successfully!"
            )

            st.subheader(
                "📚 Retrieved Information"
            )


            if not results:

                st.info(
                    "No relevant information was found."
                )

            else:

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    similarity = result["score"]


                    # -----------------------------------------
                    # RESULT CARD
                    # -----------------------------------------

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### {i}. "
                            f"{result['question']}"
                        )


                        # Similarity

                        st.progress(
                            min(
                                max(
                                    similarity,
                                    0.0
                                ),
                                1.0
                            )
                        )


                        st.caption(
                            f"Similarity Score: "
                            f"{similarity:.4f}"
                        )


                        st.markdown(
                            "**Answer:**"
                        )


                        st.write(
                            result["answer"]
                        )


            # =================================================
            # TRANSPARENCY
            # =================================================

            if results:

                st.divider()

                st.subheader(
                    "🔍 Retrieval Transparency"
                )

                st.write(
                    """
                    The system does not generate a new medical
                    answer. It converts your question into a
                    TF-IDF vector and compares it with the
                    knowledge-base vectors using cosine similarity.
                    """
                )


                transparency_df = pd.DataFrame(
                    [
                        {
                            "Rank": i,
                            "Retrieved Question":
                                result["question"],
                            "Similarity":
                                round(
                                    result["score"],
                                    4
                                )
                        }

                        for i, result in enumerate(
                            results,
                            start=1
                        )
                    ]
                )


                st.dataframe(
                    transparency_df,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as e:

            st.error(
                f"❌ Retrieval failed: {e}"
            )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Example Questions"
)

example_questions = [
    "What are the symptoms of diabetes?",
    "What causes high blood pressure?",
    "What are the symptoms of asthma?",
    "What is heart disease?",
    "How is migraine treated?"
]


for example in example_questions:

    st.write(
        f"• {example}"
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This assistant retrieves information from a medical
    question-answer dataset for educational and research
    purposes.

    The retrieved information is NOT personalized medical
    advice and should not be treated as a diagnosis or
    treatment recommendation.

    Please consult a qualified healthcare professional
    for medical concerns.
    """
)