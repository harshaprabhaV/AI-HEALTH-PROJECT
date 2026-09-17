import streamlit as st
from pathlib import Path
import json

# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

CSS_PATH = APP_DIR / "assets" / "styles.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 AI Healthcare Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered healthcare analysis and medical information system'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# FEATURES
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h2>🩺 Disease Prediction</h2>
        <p>Predict possible diseases based on selected symptoms.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h2>❤️ Heart Risk</h2>
        <p>Estimate cardiac risk using the trained machine learning model.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h2>🧠 Medical Assistant</h2>
        <p>Retrieve relevant medical information from the knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    st.markdown("""
    <div class="card">
        <h2>🏥 Hospitals & Labs</h2>
        <p>Search available hospitals and laboratory information.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="card">
        <h2>💊 Medicine Search</h2>
        <p>Search medicine information from the available dataset.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("📊 Model Information")

report_path = MODELS_DIR / "training_report.json"

if report_path.exists():

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        c1, c2, c3 = st.columns(3)

        # Disease classes
        c1.metric(
            "Disease Classes",
            report.get("disease_n_classes", "N/A")
        )

        # Disease accuracy
        accuracy = report.get("disease_test_accuracy")

        if accuracy is not None:
            c2.metric(
                "Disease Accuracy",
                f"{accuracy * 100:.2f}%"
            )
        else:
            c2.metric(
                "Disease Accuracy",
                "N/A"
            )

        # Heart accuracy
        heart_accuracy = report.get("heart_test_accuracy")

        if heart_accuracy is not None:
            c3.metric(
                "Heart Accuracy",
                f"{heart_accuracy * 100:.2f}%"
            )
        else:
            c3.metric(
                "Heart Accuracy",
                "N/A"
            )

    except Exception as e:

        st.error(
            f"Could not read training report: {e}"
        )

else:

    st.info(
        "Training report not found. "
        "Place training_report.json inside the models folder."
    )

# ============================================================
# MODEL STATUS
# ============================================================

st.divider()

st.subheader("🤖 AI System Status")

model_files = {
    "Disease Model": "disease_model.pkl",
    "Disease Label Encoder": "disease_label_encoder.pkl",
    "Heart Risk Model": "heart_risk_model.pkl",
    "Heart Risk Scaler": "heart_risk_scaler.pkl",
    "RAG Vectorizer": "rag_tfidf_vectorizer.pkl",
    "RAG Matrix": "rag_tfidf_matrix.pkl"
}

for name, filename in model_files.items():

    if (MODELS_DIR / filename).exists():
        st.success(f"✅ {name}")
    else:
        st.error(f"❌ {name} — missing")

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ Educational / research prototype only. "
    "Model predictions are not medical diagnoses. "
    "Always consult a qualified healthcare professional "
    "for medical decisions."
)