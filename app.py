import streamlit as st
import joblib
import pandas as pd

# ~~~~~ Streamlit UI Setup ~~~~~
st.set_page_config(page_title="Smart Course Enrollment Predictor", layout="centered")

st.title("Smart Course Enrollment Predictor")

@st.cache_data
def load_models():
    """
    Load the trained models and preprocessing objects from disk.
    Returns everything needed for inference:
      - linear_model, logreg_model, knn_model
      - df, scalers, label encoders
    """
    linear_model = joblib.load('linear_model.pkl')
    logreg_model = joblib.load('logreg_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    (
        df, X_lr, y_lr,
        X_logr, y_logr,
        X_knn, y_knn,
        scaler_lr, scaler_logr, scaler_knn,
        le_major, le_course
    ) = joblib.load('preprocessing_objects.pkl')

    return (
        linear_model, logreg_model, knn_model,
        df, scaler_lr, scaler_logr, scaler_knn,
        le_major, le_course
    )


# Load the models once (cached)
(
    linear_model, logreg_model, knn_model,
    df, scaler_lr, scaler_logr, scaler_knn,
    le_major, le_course
) = load_models()


def estimate_final_grade(features):
    """
    Use the Linear Regression model to predict final_grade (0-4).
    """
    features_scaled = scaler_lr.transform([features])
    return linear_model.predict(features_scaled)[0]


def predict_pass_probability(features):
    """
    Use the Logistic Regression model to predict pass probability.
    """
    features_scaled = scaler_logr.transform([features])
    pass_prob = logreg_model.predict_proba(features_scaled)[0, 1]
    return pass_prob


def find_similar_students(features, n_neighbors=3):
    """
    Use the KNN model to find n_neighbors most similar students.
    Returns a list of dicts with student info.
    """
    features_scaled = scaler_knn.transform([features])
    distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=n_neighbors)

    similar_records = []
    for idx in indices[0]:
        record = df.iloc[idx]
        similar_records.append({
            'Student ID': record['student_id'],
            'Major': record['major'],
            'Course': record['course_id'],
            'Passed': record['passed'],
            'Final Grade': record['final_grade']
        })
    return similar_records


st.write("""
Use this app to estimate how a student might perform in a particular course,
based on:
- Past GPA  
- Total credits completed  
- Weekly study hours  
- Course difficulty  
- Major  
- Course ID  

The app will:
1. Predict **final grade** using **Linear Regression**  
2. Estimate the probability of passing using **Logistic Regression**  
3. Show **similar students** (using KNN) and how they performed  
""")

# Gather unique major & course values from the dataset
all_majors = sorted(df['major'].unique())
all_courses = sorted(df['course_id'].unique())

with st.form("student_input_form"):
    st.subheader("Enter Student & Course Details")

    user_major = st.selectbox("Student Major", all_majors)
    past_gpa = st.number_input("Past GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=2.5, step=0.1)
    total_credits = st.number_input("Total Credits Completed", min_value=0, max_value=300, value=30, step=1)
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=168, value=10, step=1)

    course_id = st.selectbox("Course ID", all_courses)
    course_difficulty = st.slider("Course Difficulty (1=Easy, 10=Hard)", 1, 10, 5)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert user_major and course_id to the numeric encodings
    major_encoded = le_major.transform([user_major])[0]
    course_encoded = le_course.transform([course_id])[0]

    # The order must match how we trained:
    # [past_gpa, total_credits, study_hours, course_difficulty, major_encoded, course_encoded]
    features = [
        past_gpa,
        total_credits,
        study_hours,
        course_difficulty,
        major_encoded,
        course_encoded
    ]

    # 1) Predict final grade
    predicted_grade = estimate_final_grade(features)

    # 2) Probability of passing
    pass_probability = predict_pass_probability(features)

    # 3) Similar students
    similar_studs = find_similar_students(features, n_neighbors=3)

    # 7. Display the results
    st.success("Here are your predictions:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Final Grade (0–4)", f"{predicted_grade:.2f}")
    with col2:
        st.metric("Probability of Passing", f"{pass_probability:.2f}")

    st.write("### Top 3 Similar Students (from Historical Data)")
    st.write("These students had similar profiles and their results may give additional insight.")
    st.table(similar_studs)
