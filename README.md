# 🏥 AI Healthcare Assistant

An AI-powered healthcare assistant that combines machine learning,
medical question retrieval, and healthcare data search into a single
Streamlit application.

## 🚀 Features

### 1. Disease Prediction
Predicts a possible disease based on selected symptoms.

- Random Forest Classifier
- 41 disease classes
- 132 symptom features
- Top-3 prediction probabilities
- Disease description
- Suggested precautions
- Feature importance visualization

### 2. Heart Disease Risk Prediction

Estimates heart disease risk using clinical parameters.

Features include:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- ST slope

Technology:

- Random Forest
- StandardScaler
- Probability-based risk output

### 3. Medical Assistant

Provides medical information using a retrieval-based
question-answering system.

Pipeline:

User Question
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Top-K Relevant Medical Questions
      ↓
Retrieved Answers

The knowledge base is based on the MedQuAD Q&A dataset.

### 4. Hospitals & Labs Search

Search healthcare facilities from the available healthcare dataset.

Users can search using:

- Hospital/lab name
- Location
- City
- Other available information

### 5. Medicine Search

Search the available medicine dataset.

The application provides information from the dataset
and does not prescribe medicines.

### 6. Model Analytics

Displays information about:

- Disease model
- Heart-risk model
- RAG system
- Number of features
- Number of classes
- Model accuracy
- ROC-AUC
- Saved model artifacts

---

# 🧠 Machine Learning Architecture

## Disease Prediction

Symptoms
   ↓
Feature Vector
   ↓
Random Forest
   ↓
Disease Probabilities
   ↓
Top Predictions


## Heart Risk Prediction

Patient Parameters
   ↓
Feature Preparation
   ↓
StandardScaler
   ↓
Random Forest
   ↓
Risk Probability


## Medical Q&A

User Question
   ↓
TF-IDF
   ↓
Cosine Similarity
   ↓
Top-K Retrieval
   ↓
Medical Q&A Results

---

# 📁 Project Structure

```text
AI-Healthcare-Assistant/
│
├── notebooks/
│   └── train_and_test.ipynb
│
├── streamlit/
│   ├── app.py
│   │
│   ├── pages/
│   │   ├── 01_Disease_Prediction.py
│   │   ├── 02_Heart_Risk.py
│   │   ├── 03_Medical_Assistant.py
│   │   ├── 04_Hospitals_Labs.py
│   │   ├── 05_Medicine_Search.py
│   │   └── 06_Model_Analytics.py
│   │
│   ├── components/
│   │   ├── cards.py
│   │   ├── charts.py
│   │   └── results.py
│   │
│   ├── utils/
│   │   ├── disease_predictor.py
│   │   ├── heart_predictor.py
│   │   ├── rag_engine.py
│   │   ├── data_loader.py
│   │   ├── hospital_search.py
│   │   └── medicine_search.py
│   │
│   └── assets/
│       └── styles.css
│
├── models/
│   ├── disease_model.pkl
│   ├── disease_label_encoder.pkl
│   ├── symptom_columns.json
│   ├── symptom_importances.json
│   ├── heart_risk_model.pkl
│   ├── heart_risk_scaler.pkl
│   ├── heart_features.json
│   ├── rag_tfidf_vectorizer.pkl
│   ├── rag_tfidf_matrix.pkl
│   └── training_report.json
│
├── data/
│   ├── disease_symptom_training.csv
│   ├── disease_symptom_testing.csv
│   ├── heart_disease_cleveland_303.csv
│   ├── heart_disease_uci_1190.csv
│   ├── medquad_qa.csv
│   ├── symptom_description.csv
│   ├── symptom_precaution.csv
│   ├── symptom_severity.csv
│   ├── india_hospitals_labs.csv
│   └── india_medicine_data.csv
│
├── requirements.txt
└── README.md