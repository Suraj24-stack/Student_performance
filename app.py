import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# Use wide layout
st.set_page_config(page_title="Student Outcome & Risk Dashboard", layout="wide")

# Load model pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("student_performance_logistic_model.pkl")

try:
    model = load_pipeline()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.title("🎓 Student Academic Risk & Performance Dashboard")
st.markdown("Simulate student profiles or type custom values to forecast examination outcomes.")

# Session state initialization
cities = ["Kathmandu", "Bhaktapur", "Lalitpur", "Pokhara", "Biratnagar", "Butwal", "Dharan"]
defaults = {
    'age': 16.0,
    'gender': 'Male',
    'city': 'Kathmandu',
    'attendance': 82.0,
    'enroll_year': 2026,
    'enroll_month': 3,
    'math': 68.0,
    'science': 72.0,
    'english': 64.0,
    'social': 60.0
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Two-column layout
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.subheader(" Student Profile & Inputs")

    # 1. Preset simulation buttons outside form for instant auto-filling
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button(" Random Student", use_container_width=True):
            st.session_state.age = float(random.randint(15, 20))
            st.session_state.gender = random.choice(["Male", "Female"])
            st.session_state.city = random.choice(cities)
            st.session_state.attendance = float(random.randint(40, 95))
            st.session_state.math = float(random.randint(25, 90))
            st.session_state.science = float(random.randint(25, 90))
            st.session_state.english = float(random.randint(25, 90))
            st.session_state.social = float(random.randint(25, 90))
            st.rerun()
    with p2:
        if st.button("⚠️ At-Risk Profile", use_container_width=True):
            st.session_state.attendance = 25.0
            st.session_state.math = 22.0
            st.session_state.science = 25.0
            st.session_state.english = 30.0
            st.session_state.social = 28.0
            st.rerun()
    with p3:
        if st.button("⭐ High Performer", use_container_width=True):
            st.session_state.attendance = 95.0
            st.session_state.math = 88.0
            st.session_state.science = 90.0
            st.session_state.english = 85.0
            st.session_state.social = 84.0
            st.rerun()

    st.write("")

    # 2. Input Form with explicit Run/Submit button
    with st.form(key="student_input_form"):
        d1, d2 = st.columns(2)
        with d1:
            age = st.number_input("Age", 10.0, 25.0, key="age", step=1.0)
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            city = st.selectbox("City", cities, key="city")
        with d2:
            attendance = st.number_input("Attendance Rate (%)", 0.0, 100.0, key="attendance", step=1.0)
            enroll_year = st.selectbox("Enrollment Year", [2023, 2024, 2025, 2026], key="enroll_year")
            enroll_month = st.number_input("Enrollment Month (1-12)", 1, 12, key="enroll_month", step=1)

        st.markdown("**Subject Scores (0 - 100)**")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            math = st.number_input("Math", 0.0, 100.0, key="math", step=1.0)
        with s2:
            science = st.number_input("Science", 0.0, 100.0, key="science", step=1.0)
        with s3:
            english = st.number_input("English", 0.0, 100.0, key="english", step=1.0)
        with s4:
            social = st.number_input("Social", 0.0, 100.0, key="social", step=1.0)

        st.write("")
        # The explicit execution trigger
        submitted = st.form_submit_button(" Run Prediction & Analysis", type="primary", use_container_width=True)

with right_col:
    st.subheader("📊 Prediction & Analysis Results")

    # Construct input dataframe
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

    # Compute inferences
    prob_pass = model.predict_proba(input_data)[0][1]
    is_pass = prob_pass >= 0.50
    avg_score = round((math + science + english + social) / 4.0, 1)

    # Key metric cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Average Score", f"{avg_score}%")
    m2.metric("Attendance", f"{attendance:.1f}%")
    m3.metric("Pass Probability", f"{prob_pass:.1%}")

    st.write("")

    # Status badge
    if is_pass:
        st.success(f"### Result: Pass / Low Academic Risk\nConfidence: **{prob_pass:.1%}**")
    else:
        st.error(f"### Result: At-Risk / Academic Support Needed\nFailure Risk: **{(1 - prob_pass):.1%}**")

    st.divider()

    # Subject breakdown chart
    st.markdown("**Subject Breakdown**")
    score_df = pd.DataFrame({
        "Subject": ["Math", "Science", "English", "Social"],
        "Marks": [math, science, english, social]
    })
    st.bar_chart(score_df.set_index("Subject"))
