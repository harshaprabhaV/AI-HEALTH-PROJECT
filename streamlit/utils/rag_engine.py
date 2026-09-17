from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

STREAMLIT_DIR = CURRENT_FILE.parent.parent
PROJECT_DIR = STREAMLIT_DIR.parent

MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# LOAD TF-IDF ARTIFACTS
# ============================================================

def load_rag_model():

    vectorizer_path = (
        MODEL_DIR /
        "rag_tfidf_vectorizer.pkl"
    )

    matrix_path = (
        MODEL_DIR /
        "rag_tfidf_matrix.pkl"
    )

    dataset_path = (
        DATA_DIR /
        "medquad_qa.csv"
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    required_files = [
        vectorizer_path,
        matrix_path,
        dataset_path
    ]


    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required RAG file not found: {file_path}"
            )


    # --------------------------------------------------------
    # LOAD VECTORIZER
    # --------------------------------------------------------

    vectorizer = joblib.load(
        vectorizer_path
    )


    # --------------------------------------------------------
    # LOAD TF-IDF MATRIX
    # --------------------------------------------------------

    tfidf_matrix = joblib.load(
        matrix_path
    )


    # --------------------------------------------------------
    # LOAD MEDQUAD DATASET
    # --------------------------------------------------------

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin1"
    ]

    qa_df = None


    for encoding in encodings:

        try:

            qa_df = pd.read_csv(
                dataset_path,
                encoding=encoding
            )

            break

        except UnicodeDecodeError:

            continue


    if qa_df is None:

        raise ValueError(
            "Unable to read medquad_qa.csv"
        )


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "question",
        "answer"
    ]


    for column in required_columns:

        if column not in qa_df.columns:

            raise ValueError(
                f"Missing column '{column}' "
                f"in medquad_qa.csv"
            )


    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    qa_df = qa_df.dropna(
        subset=[
            "question",
            "answer"
        ]
    ).reset_index(
        drop=True
    )


    return (
        vectorizer,
        tfidf_matrix,
        qa_df
    )


# ============================================================
# SEARCH MEDICAL KNOWLEDGE
# ============================================================

def search_medical_qa(
    question,
    top_k=3
):

    if not question or not question.strip():

        return []


    # --------------------------------------------------------
    # LOAD RAG SYSTEM
    # --------------------------------------------------------

    (
        vectorizer,
        tfidf_matrix,
        qa_df
    ) = load_rag_model()


    # --------------------------------------------------------
    # CONVERT QUESTION TO TF-IDF
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [question.strip()]
    )


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()


    # --------------------------------------------------------
    # GET TOP RESULTS
    # --------------------------------------------------------

    top_indices = similarity_scores.argsort(
    )[::-1][:top_k]


    results = []


    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    for index in top_indices:

        # Protect against dataset/matrix mismatch

        if index >= len(qa_df):

            continue


        question_text = qa_df.iloc[
            index
        ]["question"]


        answer_text = qa_df.iloc[
            index
        ]["answer"]


        score = similarity_scores[
            index
        ]


        results.append(
            {
                "question": str(
                    question_text
                ),

                "answer": str(
                    answer_text
                ),

                "score": float(
                    score
                )
            }
        )


    return results


# ============================================================
# GET BEST ANSWER
# ============================================================

def get_best_answer(
    question
):

    results = search_medical_qa(
        question,
        top_k=1
    )


    if not results:

        return None


    return results[0]


# ============================================================
# GET KNOWLEDGE BASE SIZE
# ============================================================

def get_knowledge_base_size():

    (
        _,
        _,
        qa_df
    ) = load_rag_model()


    return len(
        qa_df
    )