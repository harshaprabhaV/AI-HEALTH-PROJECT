# MediMind AI — Healthcare Prediction System

## Overview
MediMind AI is a machine learning-based healthcare application designed to support preliminary disease prediction and heart disease risk assessment through an interactive web interface.

## Features
- **Disease Prediction:** Predicts possible diseases based on user-selected symptoms.
- **Heart Risk Assessment:** Estimates heart disease risk using a trained machine learning model.
- **Doctor Directory:** Displays available doctors and their details.
- **Appointment Booking:** Allows users to book doctor appointments.
- **Dashboard:** Displays healthcare and appointment information.
- **Dark UI:** Provides a modern interface with green-themed text.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy
- Joblib
- SQLite
- Machine Learning

## Project Structure
- `app.py` — Streamlit application
- `disease_model.pkl` — Trained disease prediction model
- `disease_label_encoder.pkl` — Disease label encoder
- `heart_risk_model.pkl` — Heart risk prediction model
- `heart_risk_scaler.pkl` — Feature scaling model
- `heart_features.json` — Heart assessment feature information
- `symptom_columns.json` — Symptom feature information
- `medical_ai.db` — SQLite database
- `requirements.txt` — Project dependencies

## Installation and Setup
1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Ensure the model files and database are available in the project directory.
4. Run the application:

```bash
streamlit run app.py
```

## Disclaimer
MediMind AI is an educational project and is not a substitute for professional medical advice, diagnosis, or treatment. Prediction results should not be used as the sole basis for healthcare decisions.

## Author
Harsha Prabha V S
