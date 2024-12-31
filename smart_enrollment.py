import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def load_and_preprocess_data(csv_file='course_enrollment_data.csv'):
    """
    Reads the CSV dataset and prepares the feature matrices for:
    1) Linear Regression (final_grade)
    2) Logistic Regression (pass/fail)
    3) KNN classification
    Returns:
       df (DataFrame),
       scaled feature matrices (X_lr, X_logr, X_knn),
       labels (y_lr, y_logr, y_knn),
       scalers, label encoders, etc.
    """
    df = pd.read_csv(csv_file)

    # Encode the 'major' column
    le_major = LabelEncoder()
    df['major_encoded'] = le_major.fit_transform(df['major'])

    # (Optional) Encode the 'course_id' if you want to treat it as a categorical feature
    le_course = LabelEncoder()
    df['course_encoded'] = le_course.fit_transform(df['course_id'])

    # Features for Linear Regression (predicting final_grade)
    features_lr = [
        'past_gpa',
        'total_credits_completed',
        'study_hours_per_week',
        'course_difficulty',
        'major_encoded'
    ]
    X_lr = df[features_lr].values
    y_lr = df['final_grade'].values

    # Features for Logistic Regression (predicting pass/fail)
    features_logr = [
        'past_gpa',
        'total_credits_completed',
        'study_hours_per_week',
        'course_difficulty',
        'major_encoded'
    ]
    X_logr = df[features_logr].values
    y_logr = df['passed'].values

    # KNN: We'll reuse the same features as logistic regression
    X_knn = X_logr
    y_knn = y_logr

    # Scale features
    scaler_lr = StandardScaler()
    X_lr_scaled = scaler_lr.fit_transform(X_lr)

    scaler_logr = StandardScaler()
    X_logr_scaled = scaler_logr.fit_transform(X_logr)

    scaler_knn = StandardScaler()
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    return (df,
            X_lr_scaled, y_lr,
            X_logr_scaled, y_logr,
            X_knn_scaled, y_knn,
            scaler_lr, scaler_logr, scaler_knn,
            le_major, le_course)

def train_models(X_lr, y_lr, X_logr, y_logr, X_knn, y_knn):
    """
    Trains:
    1) Linear Regression -> final_grade
    2) Logistic Regression -> pass/fail
    3) KNN -> pass/fail classification
    Prints basic performance metrics.
    Returns the three trained models.
    """
    # 1) Linear Regression
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )
    linear_model = LinearRegression()
    linear_model.fit(X_train_lr, y_train_lr)
    y_pred_lr = linear_model.predict(X_test_lr)
    mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
    print(f"[Linear Regression] MSE on test set: {mse_lr:.2f}")

    # 2) Logistic Regression (pass/fail)
    X_train_logr, X_test_logr, y_train_logr, y_test_logr = train_test_split(
        X_logr, y_logr, test_size=0.2, random_state=42
    )
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train_logr, y_train_logr)
    y_pred_logr = logreg_model.predict(X_test_logr)
    acc_logr = accuracy_score(y_test_logr, y_pred_logr)
    print(f"[Logistic Regression] Accuracy on test set: {acc_logr:.2f}")

    # 3) KNN
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y_knn, test_size=0.2, random_state=42
    )
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn_model.predict(X_test_knn)
    acc_knn = accuracy_score(y_test_knn, y_pred_knn)
    print(f"[KNN] Accuracy on test set (pass/fail): {acc_knn:.2f}")

    return linear_model, logreg_model, knn_model

def estimate_final_grade(linear_model, scaler_lr, features):
    """
    Given a single student's numeric features,
    predicts final grade (0-4 scale).
    """
    features_scaled = scaler_lr.transform([features])
    predicted_grade = linear_model.predict(features_scaled)[0]
    return predicted_grade

def predict_pass_fail(logreg_model, scaler_logr, features):
    """
    Given a single student's numeric features,
    returns the probability of passing the course.
    """
    features_scaled = scaler_logr.transform([features])
    # Probability vector -> [prob of 0, prob of 1]
    pass_prob = logreg_model.predict_proba(features_scaled)[0, 1]
    return pass_prob

def find_similar_students(knn_model, scaler_knn, df, features, n_neighbors=3):
    """
    Finds the n_neighbors most similar students (based on the same features),
    returns their records for analysis (e.g., did they pass or fail?).
    """
    features_scaled = scaler_knn.transform([features])  # shape (1, n_features)
    distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=n_neighbors)

    similar_records = []
    for idx in indices[0]:
        record = df.iloc[idx]
        similar_records.append({
            'student_id': record['student_id'],
            'major': record['major'],
            'passed': record['passed'],
            'final_grade': record['final_grade']
        })
    return similar_records

def main():
    # 1. Load and preprocess data
    (df,
     X_lr, y_lr,
     X_logr, y_logr,
     X_knn, y_knn,
     scaler_lr, scaler_logr, scaler_knn,
     le_major, le_course) = load_and_preprocess_data('course_enrollment_data.csv')

    # 2. Train all models
    linear_model, logreg_model, knn_model = train_models(X_lr, y_lr, X_logr, y_logr, X_knn, y_knn)

    # 3. Test the models with a "new" student's data
    # Example new student scenario:
    new_student_major = 'Physics'          # Must exist in the dataset's major column
    new_student_past_gpa = 2.9
    new_student_credits = 35
    new_student_study_hours = 7
    new_course_difficulty = 8.0            # Suppose the course is quite challenging

    # Convert major to numeric
    major_encoded = le_major.transform([new_student_major])[0]

    # Build feature vector for LR & Logistic:
    # [past_gpa, total_credits_completed, study_hours_per_week, course_difficulty, major_encoded]
    new_features = [
        new_student_past_gpa,
        new_student_credits,
        new_student_study_hours,
        new_course_difficulty,
        major_encoded
    ]

    # Predict final grade
    predicted_grade = estimate_final_grade(linear_model, scaler_lr, new_features)
    print(f"\n[Example Student] Predicted Final Grade: {predicted_grade:.2f}")

    # Predict pass probability
    pass_probability = predict_pass_fail(logreg_model, scaler_logr, new_features)
    print(f"[Example Student] Probability of Passing: {pass_probability:.2f}")

    # Find similar students
    similar_studs = find_similar_students(knn_model, scaler_knn, df, new_features, n_neighbors=3)
    print("\n[Example Student] Top 3 Similar Students:")
    for s in similar_studs:
        print(f"  StudentID={s['student_id']} | "
              f"Major={s['major']} | "
              f"Passed={s['passed']} | "
              f"FinalGrade={s['final_grade']:.2f}")

    # 4. Optionally save your trained models with joblib for later use (e.g., Flask app)
    joblib.dump(linear_model, 'linear_model.pkl')
    joblib.dump(logreg_model, 'logreg_model.pkl')
    joblib.dump(knn_model, 'knn_model.pkl')
    # Save scalers, encoders, and the raw dataframe for reference
    joblib.dump((df, X_lr, y_lr, X_logr, y_logr, X_knn, y_knn,
                 scaler_lr, scaler_logr, scaler_knn, le_major, le_course),
                'preprocessing_objects.pkl')

if __name__ == "__main__":
    main()
