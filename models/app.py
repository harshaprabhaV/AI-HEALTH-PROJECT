import json
import sqlite3
from datetime import date, datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="MediMind AI | Healthcare Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "medical_ai.db"

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_assets():
    required = [
        "disease_model.pkl",
        "disease_label_encoder.pkl",
        "symptom_columns.json",
        "heart_risk_model.pkl",
        "heart_risk_scaler.pkl",
        "heart_features.json",
        "rag_tfidf_vectorizer.pkl",
        "rag_tfidf_matrix.pkl",
    ]
    missing = [name for name in required if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files in the same folder as app.py: " + ", ".join(missing)
        )

    disease_model = joblib.load(BASE_DIR / "disease_model.pkl")
    disease_encoder = joblib.load(BASE_DIR / "disease_label_encoder.pkl")
    with open(BASE_DIR / "symptom_columns.json", encoding="utf-8") as file:
        symptoms = json.load(file)

    heart_model = joblib.load(BASE_DIR / "heart_risk_model.pkl")
    heart_scaler = joblib.load(BASE_DIR / "heart_risk_scaler.pkl")
    with open(BASE_DIR / "heart_features.json", encoding="utf-8") as file:
        heart_features = json.load(file)

    vectorizer = joblib.load(BASE_DIR / "rag_tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load(BASE_DIR / "rag_tfidf_matrix.pkl")

    if hasattr(disease_model, "n_features_in_") and len(symptoms) != disease_model.n_features_in_:
        raise ValueError("symptom_columns.json does not match disease_model.pkl.")
    if hasattr(heart_model, "n_features_in_") and len(heart_features) != heart_model.n_features_in_:
        raise ValueError("heart_features.json does not match heart_risk_model.pkl.")
    if hasattr(heart_scaler, "n_features_in_") and len(heart_features) != heart_scaler.n_features_in_:
        raise ValueError("heart_features.json does not match heart_risk_scaler.pkl.")

    return {
        "disease_model": disease_model,
        "disease_encoder": disease_encoder,
        "symptoms": symptoms,
        "heart_model": heart_model,
        "heart_scaler": heart_scaler,
        "heart_features": heart_features,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }


try:
    assets = load_assets()
    MODEL_ERROR = None
except Exception as error:
    assets = None
    MODEL_ERROR = str(error)

# =========================================================
# DATABASE HELPERS
# =========================================================
def connect_db():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH.name}. Put medical_ai.db beside app.py."
        )
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def table_exists(connection, table_name):
    result = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return result is not None


def get_database_counts():
    try:
        with connect_db() as connection:
            counts = {}
            for table in ("patients", "doctors", "appointments"):
                if not table_exists(connection, table):
                    return None
                counts[table] = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            return counts
    except (sqlite3.Error, FileNotFoundError):
        return None


def get_doctors():
    with connect_db() as connection:
        if not table_exists(connection, "doctors"):
            raise RuntimeError("The doctors table is missing from medical_ai.db.")
        return pd.read_sql_query(
            """
            SELECT id, full_name, specialization, hospital_name, city, state,
                   years_experience, consultation_fee, available_days
            FROM doctors
            ORDER BY full_name
            """,
            connection,
        )


def create_appointment(patient_name, email, phone, age, gender,
                       doctor_id, appointment_date, appointment_time, reason):
    """Create/find a patient and save an appointment in the existing SQLite schema."""
    with connect_db() as connection:
        for table in ("patients", "doctors", "appointments"):
            if not table_exists(connection, table):
                raise RuntimeError(f"The {table} table is missing from medical_ai.db.")

        patient = connection.execute(
            "SELECT id FROM patients WHERE lower(email)=lower(?) LIMIT 1",
            (email.strip(),),
        ).fetchone()

        if patient:
            patient_id = patient[0]
            connection.execute(
                """
                UPDATE patients
                SET full_name=?, phone=?, age=?, gender=?
                WHERE id=?
                """,
                (patient_name.strip(), phone.strip(), int(age), gender, patient_id),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO patients
                    (full_name, email, phone, age, gender, medical_history, created_at)
                VALUES (?, ?, ?, ?, ?, '', ?)
                """,
                (
                    patient_name.strip(), email.strip(), phone.strip(),
                    int(age), gender, datetime.now().isoformat(timespec="seconds"),
                ),
            )
            patient_id = cursor.lastrowid

        # Prevent double-booking the same doctor at the same date/time.
        conflict = connection.execute(
            """
            SELECT 1 FROM appointments
            WHERE doctor_id=? AND appointment_date=? AND appointment_time=?
              AND lower(status) NOT IN ('cancelled', 'canceled')
            LIMIT 1
            """,
            (int(doctor_id), appointment_date.isoformat(), appointment_time.strftime("%H:%M")),
        ).fetchone()
        if conflict:
            raise ValueError(
                "This doctor already has an appointment at that date and time. "
                "Please choose another slot."
            )

        cursor = connection.execute(
            """
            INSERT INTO appointments
                (patient_id, doctor_id, appointment_date, appointment_time,
                 reason, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                int(doctor_id),
                appointment_date.isoformat(),
                appointment_time.strftime("%H:%M"),
                reason.strip(),
                "scheduled",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        appointment_id = cursor.lastrowid
        connection.commit()
        return appointment_id


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #080D14;
        color: #F1F5F9;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1450px;
        padding: 1.2rem 4% 3rem;
    }
    #MainMenu, footer { visibility: hidden; }
    .navbar {
        background: #111923;
        border: 1px solid #263443;
        border-radius: 18px;
        padding: 17px 23px;
        margin-bottom: 22px;
    }
    .brand {
        color: #5EEAD4;
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -.7px;
    }
    .nav-subtitle {
        color: #91A4B7;
        font-size: 11px;
        letter-spacing: 1.3px;
    }
    .hero {
        background: linear-gradient(120deg, #102D32, #111827 75%);
        border: 1px solid #23454A;
        border-radius: 24px;
        padding: 34px 30px;
        min-height: 300px;
    }
    .hero h1 {
        font-size: clamp(32px, 4vw, 47px);
        color: #F8FAFC;
        line-height: 1.12;
        margin: 12px 0;
    }
    .hero p {
        color: #B0C0CC;
        font-size: 16px;
        line-height: 1.8;
    }
    .eyebrow {
        color: #5EEAD4;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 800;
    }
    .section-title {
        color: #F1F5F9;
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
    }
    .feature-card {
        background: #111923;
        border: 1px solid #263443;
        border-radius: 18px;
        padding: 23px;
        min-height: 155px;
    }
    .feature-card h3 { color: #5EEAD4; font-size: 19px; }
    .feature-card p { color: #A8B7C7; line-height: 1.65; font-size: 14px; }
    .metric-card {
        background: #111923;
        border: 1px solid #263443;
        border-radius: 16px;
        padding: 20px;
    }
    .metric-label { color: #91A4B7; font-size: 13px; }
    .metric-value { color: #5EEAD4; font-size: 27px; font-weight: 800; }
    .stButton > button, .stFormSubmitButton > button {
        background: #5EEAD4;
        color: #081116;
        border: 0;
        border-radius: 10px;
        font-weight: 750;
        min-height: 42px;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #99F6E4;
        color: #081116;
        border: 0;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"],
    .stDateInput input, .stTimeInput input {
        background-color: #111923;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #263443; }
    .footer {
        color: #718096;
        text-align: center;
        padding: 24px 0 5px;
        font-size: 12px;
    }

    /* High-contrast green text throughout the entire app */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1,
    .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp li, .stApp small, .stApp strong, .stApp em,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stCaptionContainer"], [data-testid="stAlert"] p,
    [data-testid="stDataFrame"] *, [data-testid="stTable"] *,
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stDateInput input, .stTimeInput input,
    [data-baseweb="select"] *, [data-baseweb="popover"] *,
    [data-testid="stRadio"] *, [data-testid="stCheckbox"] *,
    [data-testid="stMultiSelect"] *, .footer {
        color: #5EEAD4 !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #7DD3C7 !important;
        opacity: 1;
    }
    .stButton > button, .stFormSubmitButton > button {
        background: #111923;
        color: #5EEAD4 !important;
        border: 1px solid #2DD4BF;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #18302F;
        color: #99F6E4 !important;
        border: 1px solid #5EEAD4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# NAVBAR + BUTTON NAVIGATION (NO RADIO BUTTONS)
# =========================================================
PAGES = [
    "Home",
    "Disease Prediction",
    "Heart Risk",
    "Book Appointment",
    "Dashboard",
]
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "Home"


def go_to(page_name):
    st.session_state.page = page_name


st.markdown(
    """
    <div class="navbar">
        <div class="brand">✚ MediMind AI</div>
        <div class="nav-subtitle">INTELLIGENT HEALTHCARE · MACHINE LEARNING</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(PAGES), gap="small")
for index, page_name in enumerate(PAGES):
    with nav_cols[index]:
        button_label = ("● " if st.session_state.page == page_name else "") + page_name
        if st.button(button_label, key=f"nav_{index}", use_container_width=True):
            go_to(page_name)
            st.rerun()

st.write("")
page = st.session_state.page

# =========================================================
# HOME
# =========================================================
if page == "Home":
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero">
                <div class="eyebrow">AI-POWERED HEALTHCARE</div>
                <h1>Your Health.<br>Smarter Insights.</h1>
                <p>
                    Explore symptom-based disease prediction, heart-risk
                    assessment, and doctor appointments in one intelligent
                    healthcare platform.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("Explore Disease AI →", use_container_width=True):
                go_to("Disease Prediction")
                st.rerun()
        with cta2:
            if st.button("Book Appointment →", use_container_width=True):
                go_to("Book Appointment")
                st.rerun()

    with right:
        st.image(
            "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d"
            "?auto=format&fit=crop&w=1200&q=85",
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-title">Your AI Health Toolkit</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧬 Disease Prediction</h3>
                <p>Explore symptom patterns with your trained disease model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Disease Prediction", key="home_disease", use_container_width=True):
            go_to("Disease Prediction")
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>❤️ Heart Risk</h3>
                <p>Explore model estimates using the saved heart-risk model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Heart Risk", key="home_heart", use_container_width=True):
            go_to("Heart Risk")
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📅 Book Appointment</h3>
                <p>Choose a doctor, date, and time and save an appointment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Find a Doctor", key="home_booking", use_container_width=True):
            go_to("Book Appointment")
            st.rerun()

    st.write("")
    st.info(
        "MediMind AI provides experimental model estimates and medical information, "
        "not a diagnosis. For medical decisions, consult a qualified healthcare professional."
    )

# =========================================================
# DISEASE PREDICTION
# =========================================================
elif page == "Disease Prediction":
    st.title("🧬 Disease Prediction")
    st.caption("Select the symptoms you are experiencing.")

    if assets is None:
        st.error(f"Could not load models: {MODEL_ERROR}")
        st.stop()

    symptom_columns = assets["symptoms"]

    def symptom_label(symptom):
        return symptom.replace("_", " ").replace(".", " ").title()

    symptom_options = {symptom_label(s): s for s in symptom_columns}
    selected_labels = st.multiselect(
        "Search and select symptoms",
        options=sorted(symptom_options.keys()),
        placeholder="Type a symptom, such as fatigue or headache",
    )
    st.caption(f"{len(selected_labels)} symptom(s) selected")

    if st.button("Analyze Symptoms", use_container_width=True):
        if not selected_labels:
            st.warning("Please select at least one symptom.")
        else:
            selected_raw = {symptom_options[label] for label in selected_labels}
            values = [1 if symptom in selected_raw else 0 for symptom in symptom_columns]
            row = pd.DataFrame([values], columns=symptom_columns)

            try:
                model = assets["disease_model"]
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(row)[0]
                    classes = list(model.classes_)
                    top_indices = np.argsort(probabilities)[::-1][:3]
                    results = []
                    for idx in top_indices:
                        encoded_class = classes[int(idx)]
                        try:
                            disease = assets["disease_encoder"].inverse_transform(
                                [encoded_class]
                            )[0]
                        except Exception:
                            disease = str(encoded_class)
                        results.append((str(disease), float(probabilities[int(idx)])))

                    st.markdown("### Model results")
                    st.subheader(results[0][0])
                    st.caption(
                        f"Model score: {results[0][1] * 100:.1f}% "
                        "(not a clinically verified probability)."
                    )
                    st.markdown("#### Other model matches")
                    for disease, score in results:
                        st.write(f"**{disease}**")
                        st.progress(float(np.clip(score, 0, 1)))
                        st.caption(f"Model score: {score * 100:.1f}%")
                else:
                    predicted = model.predict(row)[0]
                    try:
                        predicted = assets["disease_encoder"].inverse_transform([predicted])[0]
                    except Exception:
                        predicted = str(predicted)
                    st.subheader(f"Predicted class: {predicted}")

                st.warning(
                    "This is an experimental machine-learning result, not a diagnosis. "
                    "Seek medical advice if symptoms persist, worsen, or concern you."
                )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

# =========================================================
# HEART RISK
# =========================================================
elif page == "Heart Risk":
    st.title("❤️ Heart Risk Assessment")
    st.caption("Enter measurements using the same coding as your training dataset.")

    if assets is None:
        st.error(f"Could not load models: {MODEL_ERROR}")
        st.stop()

    with st.form("heart_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", min_value=18, max_value=110, value=45)
            sex = st.selectbox(
                "Sex (dataset code)",
                [0, 1],
                format_func=lambda x: "0 — Female" if x == 0 else "1 — Male",
            )
            chest_pain_type = st.selectbox("Chest pain type (dataset code)", [1, 2, 3, 4])
            resting_bp_s = st.number_input("Resting blood pressure (mmHg)", 50, 250, 120)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 50, 700, 200)
        with right:
            resting_ecg = st.selectbox("Resting ECG (dataset code)", [0, 1, 2])
            max_heart_rate = st.number_input("Maximum heart rate (bpm)", 50, 250, 150)
            exercise_angina = st.selectbox(
                "Exercise-induced angina",
                [0, 1],
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            )
            oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 0.0, step=0.1)
            st_slope = st.selectbox("ST slope (dataset code)", [1, 2, 3])

        submitted = st.form_submit_button("Calculate Model Estimate", use_container_width=True)

    if submitted:
        try:
            values = {
                "age": age,
                "sex": sex,
                "chest_pain_type": chest_pain_type,
                "resting_bp_s": resting_bp_s,
                "cholesterol": cholesterol,
                "resting_ecg": resting_ecg,
                "max_heart_rate": max_heart_rate,
                "exercise_angina": exercise_angina,
                "oldpeak": oldpeak,
                "ST_slope": st_slope,
            }
            features = assets["heart_features"]
            row = pd.DataFrame([[values[name] for name in features]], columns=features)
            scaled = assets["heart_scaler"].transform(row)
            model = assets["heart_model"]
            prediction = int(model.predict(scaled)[0])

            st.markdown("### Model result")
            if hasattr(model, "predict_proba") and 1 in list(model.classes_):
                positive_index = list(model.classes_).index(1)
                score = float(model.predict_proba(scaled)[0][positive_index])
                st.metric("Model score for class 1", f"{score * 100:.1f}%")

            if prediction == 1:
                st.error(
                    "The model classified this input as class 1. "
                    "Please discuss your measurements with a healthcare professional."
                )
            else:
                st.success(
                    "The model classified this input as class 0. "
                    "This does not rule out heart disease."
                )
            st.caption(
                "Class labels and scores reflect the training dataset and are not a "
                "clinical diagnosis or a validated personal risk estimate."
            )
        except Exception as error:
            st.error(f"Heart-risk prediction failed: {error}")

# =========================================================
# BOOK APPOINTMENT
# =========================================================
elif page == "Book Appointment":
    st.title("📅 Book an Appointment")
    st.caption("Choose a doctor and appointment slot. Your booking is saved to medical_ai.db.")

    try:
        doctors = get_doctors()
    except Exception as error:
        doctors = pd.DataFrame()
        st.error(f"Could not load doctors: {error}")

    if doctors.empty:
        st.info("No doctors are available in the database yet.")
    else:
        doctor_labels = {}
        for _, doctor in doctors.iterrows():
            label = (
                f"{doctor['full_name']} — {doctor['specialization']} "
                f"({doctor['city']}, {doctor['state']})"
            )
            doctor_labels[label] = doctor

        with st.form("appointment_form", clear_on_submit=False):
            st.markdown("### Patient details")
            c1, c2 = st.columns(2)
            with c1:
                patient_name = st.text_input("Full name *", max_chars=150)
                email = st.text_input("Email address *", max_chars=150)
                phone = st.text_input("Phone number *", max_chars=20)
            with c2:
                age = st.number_input("Age *", min_value=1, max_value=120, value=25)
                gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Other"])

            st.markdown("### Doctor and schedule")
            selected_doctor_label = st.selectbox("Select doctor *", list(doctor_labels.keys()))
            selected_doctor = doctor_labels[selected_doctor_label]

            fee = selected_doctor.get("consultation_fee")
            experience = selected_doctor.get("years_experience")
            hospital = selected_doctor.get("hospital_name")
            available_days = selected_doctor.get("available_days")

            info_cols = st.columns(3)
            with info_cols[0]:
                st.caption(f"Consultation fee: ₹{fee:g}" if pd.notna(fee) else "Consultation fee: Not listed")
            with info_cols[1]:
                st.caption(f"Experience: {int(experience)} years" if pd.notna(experience) else "Experience: Not listed")
            with info_cols[2]:
                st.caption(f"Available days: {available_days}" if pd.notna(available_days) else "Availability: Confirm with clinic")
            if pd.notna(hospital):
                st.caption(f"Hospital: {hospital}")

            date_col, time_col = st.columns(2)
            with date_col:
                appointment_date = st.date_input(
                    "Appointment date *",
                    value=date.today(),
                    min_value=date.today(),
                )
            with time_col:
                appointment_time = st.time_input(
                    "Appointment time *",
                    value=time(10, 0),
                    step=1800,
                )

            reason = st.text_area(
                "Reason for visit",
                placeholder="Briefly describe the reason for your appointment",
                max_chars=1000,
            )
            submitted = st.form_submit_button("Confirm Appointment", use_container_width=True)

        if submitted:
            if not patient_name.strip() or not email.strip() or not phone.strip():
                st.error("Please complete your name, email address, and phone number.")
            elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                st.error("Please enter a valid email address.")
            elif appointment_date < date.today():
                st.error("Choose today or a future date.")
            else:
                try:
                    appointment_id = create_appointment(
                        patient_name=patient_name,
                        email=email,
                        phone=phone,
                        age=age,
                        gender=gender,
                        doctor_id=int(selected_doctor["id"]),
                        appointment_date=appointment_date,
                        appointment_time=appointment_time,
                        reason=reason,
                    )
                    st.success(f"Appointment booked successfully! Booking ID: #{appointment_id}")
                    st.write(f"**Patient:** {patient_name}")
                    st.write(f"**Doctor:** {selected_doctor['full_name']}")
                    st.write(
                        f"**Date and time:** {appointment_date.strftime('%d %B %Y')} "
                        f"at {appointment_time.strftime('%I:%M %p')}"
                    )
                    st.info(
                        "This is a local database booking. Please confirm the appointment "
                        "with the clinic; this app does not send email or SMS notifications."
                    )
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Could not save appointment: {error}")

    st.markdown("### Existing appointments")
    try:
        with connect_db() as connection:
            appointments = pd.read_sql_query(
                """
                SELECT a.id AS booking_id, p.full_name AS patient,
                       d.full_name AS doctor, d.specialization,
                       a.appointment_date, a.appointment_time,
                       a.reason, a.status
                FROM appointments a
                JOIN patients p ON p.id = a.patient_id
                JOIN doctors d ON d.id = a.doctor_id
                ORDER BY a.appointment_date DESC, a.appointment_time DESC
                LIMIT 50
                """,
                connection,
            )
        if appointments.empty:
            st.caption("No appointments booked yet.")
        else:
            st.dataframe(appointments, use_container_width=True, hide_index=True)
    except Exception as error:
        st.caption(f"Appointment history is unavailable: {error}")

# =========================================================
# DASHBOARD
# =========================================================
elif page == "Dashboard":
    st.title("📊 Healthcare Dashboard")
    st.caption("Overview of records in the local SQLite database.")
    counts = get_database_counts()

    if counts is None:
        st.warning(
            "Could not read medical_ai.db or expected tables are missing. "
            "Place the correct database beside app.py."
        )
    else:
        c1, c2, c3 = st.columns(3)
        for col, label, key in [
            (c1, "Registered patients", "patients"),
            (c2, "Doctors", "doctors"),
            (c3, "Appointments", "appointments"),
        ]:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{counts[key]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Doctor directory")
        try:
            doctors = get_doctors()
            if doctors.empty:
                st.info("No doctor records are available.")
            else:
                display_columns = [
                    "full_name", "specialization", "hospital_name",
                    "city", "state", "years_experience", "consultation_fee",
                    "available_days",
                ]
                st.dataframe(
                    doctors[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as error:
            st.error(f"Could not load doctor records: {error}")

        st.markdown("### Recent appointments")
        try:
            with connect_db() as connection:
                recent = pd.read_sql_query(
                    """
                    SELECT a.id AS booking_id, p.full_name AS patient,
                           d.full_name AS doctor, d.specialization,
                           a.appointment_date, a.appointment_time, a.status
                    FROM appointments a
                    JOIN patients p ON p.id = a.patient_id
                    JOIN doctors d ON d.id = a.doctor_id
                    ORDER BY a.id DESC LIMIT 10
                    """,
                    connection,
                )
            st.dataframe(recent, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Could not load recent appointments: {error}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        MEDIMIND AI · INTELLIGENT HEALTHCARE<br>
        For educational and informational purposes only. Not a substitute for professional medical care.
    </div>
    """,
    unsafe_allow_html=True,
)
