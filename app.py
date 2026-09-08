import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Outcome Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Enter student information to evaluate academic risk and predict pass probability.")

# 1. Load trained model pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("student_performance_logistic_model.pkl")

try:
    model = load_pipeline()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. User Input Forms
st.subheader("Student Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10.0, max_value=25.0, value=16.0, step=1.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city = st.selectbox("City", ["Kathmandu", "Bhaktapur", "Lalitpur", "Pokhara", "Biratnagar", "Butwal", "Dharan"])
    attendance = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)

with col2:
    math = st.slider("Math Marks", 0.0, 100.0, 65.0)
    science = st.slider("Science Marks", 0.0, 100.0, 65.0)
    english = st.slider("English Marks", 0.0, 100.0, 65.0)
    social = st.slider("Social Marks", 0.0, 100.0, 65.0)

enroll_col1, enroll_col2 = st.columns(2)
with enroll_col1:
    enroll_year = st.selectbox("Enrollment Year", [2023, 2024, 2025, 2026], index=0)
with enroll_col2:
    enroll_month = st.slider("Enrollment Month", 1, 12, 3)

# 3. Predict Button
if st.button("Predict Student Outcome", type="primary"):
    input_data = pd.DataFrame([{
        'Age': age,
        'Math': math,
        'Science': science,
        'English': english,
        'Social': social,
        'Attendance': attendance,
        'Enrollment_Year': enroll_year,
        'Enrollment_Month': enroll_month,
        'Gender': gender,
        'City': city
    }])

    prob_pass = model.predict_proba(input_data)[0][1]
    prediction = int(prob_pass >= 0.50)

    st.divider()
    st.subheader("Prediction Result")

    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Predicted Pass Probability", f"{prob_pass:.1%}")

    if prediction == 1:
        col_metric2.success("Status: Pass / Low Risk")
    else:
        col_metric2.error("Status: At-Risk / High Attention Needed")
