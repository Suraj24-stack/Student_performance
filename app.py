import streamlit as st
import pandas as pd
import joblib
import random

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

# 2. State initialization for Manual vs Auto-fill
cities = ["Kathmandu", "Bhaktapur", "Lalitpur", "Pokhara", "Biratnagar", "Butwal", "Dharan"]

defaults = {
    'age': 16.0,
    'gender': 'Male',
    'city': 'Kathmandu',
    'attendance': 80.0,
    'enroll_year': 2026,
    'enroll_month': 3,
    'math': 65.0,
    'science': 65.0,
    'english': 65.0,
    'social': 65.0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Auto-fill / Quick Preset buttons
st.subheader("Presets & Controls")
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    if st.button(" Auto-Fill Random Student"):
        st.session_state.age = float(random.randint(15, 20))
        st.session_state.gender = random.choice(["Male", "Female"])
        st.session_state.city = random.choice(cities)
        st.session_state.attendance = round(random.uniform(40.0, 95.0), 1)
        st.session_state.math = round(random.uniform(25.0, 95.0), 1)
        st.session_state.science = round(random.uniform(25.0, 95.0), 1)
        st.session_state.english = round(random.uniform(25.0, 95.0), 1)
        st.session_state.social = round(random.uniform(25.0, 95.0), 1)
        st.rerun()

with btn_col2:
    if st.button("⚠️ Fill At-Risk Student"):
        st.session_state.attendance = 25.0
        st.session_state.math = 20.0
        st.session_state.science = 22.0
        st.session_state.english = 30.0
        st.session_state.social = 28.0
        st.rerun()

with btn_col3:
    if st.button("⭐ Fill High Performer"):
        st.session_state.attendance = 92.0
        st.session_state.math = 88.0
        st.session_state.science = 85.0
        st.session_state.english = 90.0
        st.session_state.social = 86.0
        st.rerun()

st.divider()

# 3. Form Inputs
st.subheader("Student Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10.0, max_value=25.0, key="age", step=1.0)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    city = st.selectbox("City", cities, key="city")
    attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, key="attendance", step=1.0)
    enroll_year = st.selectbox("Enrollment Year", [2023, 2024, 2025, 2026], key="enroll_year")
    enroll_month = st.number_input("Enrollment Month", min_value=1, max_value=12, key="enroll_month", step=1)

with col2:
    math = st.number_input("Math Marks (0-100)", min_value=0.0, max_value=100.0, key="math", step=1.0)
    science = st.number_input("Science Marks (0-100)", min_value=0.0, max_value=100.0, key="science", step=1.0)
    english = st.number_input("English Marks (0-100)", min_value=0.0, max_value=100.0, key="english", step=1.0)
    social = st.number_input("Social Marks (0-100)", min_value=0.0, max_value=100.0, key="social", step=1.0)

# 4. Predict Button
st.write("")
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
