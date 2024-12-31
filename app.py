import streamlit as st
import joblib
import pandas as pd

# streamlit UI
st.set_page_config(page_title="Smart Course Enrollment Predictor", layout="centered")

st.title("Smart Course Enrollment Predictor")


#Load the trained models and preprocessing objects
@st.cache_data  # Cache so we don't reload on every interaction
def load_models():
    linear_model = joblib.load('linear_model.pkl')
    logreg_model = joblib.load('logreg_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    (df, X_lr, y_lr, X_logr, y_logr, X_knn, y_knn,
     scaler_lr, scaler_logr, scaler_knn,
     le_major, le_course) = joblib.load('preprocessing_objects.pkl')
    return (linear_model, logreg_model, knn_model,
            df, scaler_lr, scaler_logr, scaler_knn,
            le_major, le_course)


linear_model, logreg_model, knn_model, \
df, scaler_lr, scaler_logr, scaler_knn, \
le_major, le_course = load_models()



def estimate_final_grade(features):
    """Given a single student's numeric features, predicts final grade."""
    features_scaled = scaler_lr.transform([features])
    predicted_grade = linear_model.predict(features_scaled)[0]
    return predicted_grade


def predict_pass_probability(features):
    """Given a single student's numeric features, returns pass probability."""
    features_scaled = scaler_logr.transform([features])
    pass_prob = logreg_model.predict_proba(features_scaled)[0, 1]
    return pass_prob


def find_similar_students(features, n_neighbors=3):
    """Find the n_neighbors most similar students in df using KNN."""
    features_scaled = scaler_knn.transform([features])
    distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=n_neighbors)

    similar_records = []
    for idx in indices[0]:
        record = df.iloc[idx]
        similar_records.append({
            'Student ID': record['student_id'],
            'Major': record['major'],
            'Passed': record['passed'],
            'Final Grade': record['final_grade']
        })
    return similar_records



st.write("""
This app suggests how a student might perform in a given course, 
helping them make informed enrollment decisions. It uses three models:
- **Linear Regression**: Predicts potential final grade (0–4 scale).
- **Logistic Regression**: Estimates probability of passing (0–1).
- **KNN**: Finds similar students and shows their outcomes.
""")

# 4. Create a form to collect user input
majors_available = list(df['major'].unique())

with st.form("student_input_form"):
    st.subheader("Enter Student & Course Details")
    user_major = st.selectbox("Student Major", majors_available)
    past_gpa = st.number_input("Past GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=2.5, step=0.1)
    total_credits = st.number_input("Total Credits Completed", min_value=0, max_value=200, value=30, step=1)
    study_hours = st.number_input("Study Hours per Week", min_value=0, max_value=70, value=10, step=1)
    course_difficulty = st.slider("Course Difficulty (1=Easy, 10=Hard)", min_value=1, max_value=10, value=5)

    # Optional: If you want to reflect actual courses from the data
    # course_id = st.selectbox("Course ID", list(df['course_id'].unique()))

    # Pressing "Submit" triggers a form submission
    submitted = st.form_submit_button("Predict")

if submitted:
    # 5. Convert user_major to label-encoded integer
    major_encoded = le_major.transform([user_major])[0]

    # Build feature vector: [past_gpa, credits, study_hours, difficulty, major_encoded]
    features = [past_gpa, total_credits, study_hours, course_difficulty, major_encoded]

    # 6. Model Predictions
    predicted_grade = estimate_final_grade(features)
    pass_probability = predict_pass_probability(features)
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
