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
    Reads 'course_enrollment_data.csv' and returns:
      - DataFrame
      - Feature matrices for (LR, Logistic, KNN)
      - Labels (y_lr, y_logr, y_knn)
      - Scalers for each feature matrix
      - LabelEncoders for 'major' and 'course_id'

    Expects columns:
      [student_id, major, past_gpa, total_credits_completed,
       study_hours_per_week, course_id, course_difficulty,
       final_grade, passed]
    """
    df = pd.read_csv(csv_file)

    # 1) Encode the 'major' column
    le_major = LabelEncoder()
    df['major_encoded'] = le_major.fit_transform(df['major'])

    # 2) Encode the 'course_id' column
    le_course = LabelEncoder()
    df['course_encoded'] = le_course.fit_transform(df['course_id'])

    # ~~~~~~~~~~~~~~ Feature Engineering ~~~~~~~~~~~~~~
    # We'll include 'course_encoded' as a feature in all 3 models:
    #   - Linear Regression => predict final_grade
    #   - Logistic Regression => predict pass/fail
    #   - KNN => classification for pass/fail or similarity

    # Features for Linear Regression
    features_lr = [
        'past_gpa',
        'total_credits_completed',
        'study_hours_per_week',
        'course_difficulty',
        'major_encoded',
        'course_encoded'
    ]
    X_lr = df[features_lr].values
    y_lr = df['final_grade'].values  # 0-4 scale

    # Features for Logistic Regression
    features_logr = [
        'past_gpa',
        'total_credits_completed',
        'study_hours_per_week',
        'course_difficulty',
        'major_encoded',
        'course_encoded'
    ]
    X_logr = df[features_logr].values
    y_logr = df['passed'].values  # 1 if passed, 0 if failed

    # KNN uses the same features (pass/fail classification or similarity measure)
    X_knn = X_logr
    y_knn = y_logr

    # ~~~~~~~~~~~~~~ Scaling ~~~~~~~~~~~~~~
    scaler_lr = StandardScaler()
    X_lr_scaled = scaler_lr.fit_transform(X_lr)

    scaler_logr = StandardScaler()
    X_logr_scaled = scaler_logr.fit_transform(X_logr)

    scaler_knn = StandardScaler()
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    return (
        df,
        X_lr_scaled, y_lr,
        X_logr_scaled, y_logr,
        X_knn_scaled, y_knn,
        scaler_lr, scaler_logr, scaler_knn,
        le_major, le_course
    )


def train_models(X_lr, y_lr, X_logr, y_logr, X_knn, y_knn):
    """
    Trains three models and prints simple metrics:
      - Linear Regression -> final_grade
      - Logistic Regression -> pass/fail
      - KNN -> pass/fail
    Returns the trained models.
    """

    # ~~~~~~~~~~~~~~ Linear Regression ~~~~~~~~~~~~~~
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )
    linear_model = LinearRegression()
    linear_model.fit(X_train_lr, y_train_lr)

    y_pred_lr = linear_model.predict(X_test_lr)
    mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
    print(f"[Linear Regression] Test MSE: {mse_lr:.2f}")

    # ~~~~~~~~~~~~~~ Logistic Regression ~~~~~~~~~~~~~~
    X_train_logr, X_test_logr, y_train_logr, y_test_logr = train_test_split(
        X_logr, y_logr, test_size=0.2, random_state=42
    )
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train_logr, y_train_logr)

    y_pred_logr = logreg_model.predict(X_test_logr)
    acc_logr = accuracy_score(y_test_logr, y_pred_logr)
    print(f"[Logistic Regression] Test Accuracy: {acc_logr:.2f}")

    # ~~~~~~~~~~~~~~ KNN ~~~~~~~~~~~~~~
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y_knn, test_size=0.2, random_state=42
    )
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_knn, y_train_knn)

    y_pred_knn = knn_model.predict(X_test_knn)
    acc_knn = accuracy_score(y_test_knn, y_pred_knn)
    print(f"[KNN] Test Accuracy: {acc_knn:.2f}")

    return linear_model, logreg_model, knn_model


def main():
    # 1) Load + preprocess
    (df,
     X_lr, y_lr,
     X_logr, y_logr,
     X_knn, y_knn,
     scaler_lr, scaler_logr, scaler_knn,
     le_major, le_course) = load_and_preprocess_data('course_enrollment_data.csv')

    # 2) Train the models
    linear_model, logreg_model, knn_model = train_models(X_lr, y_lr, X_logr, y_logr, X_knn, y_knn)

    # 3) Save models + objects to disk
    joblib.dump(linear_model, 'linear_model.pkl')
    joblib.dump(logreg_model, 'logreg_model.pkl')
    joblib.dump(knn_model, 'knn_model.pkl')
    joblib.dump((
        df,
        X_lr, y_lr,
        X_logr, y_logr,
        X_knn, y_knn,
        scaler_lr, scaler_logr, scaler_knn,
        le_major, le_course
    ), 'preprocessing_objects.pkl')

    print("\nModels and preprocessing objects saved to disk!")


if __name__ == "__main__":
    main()
