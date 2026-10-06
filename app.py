import joblib
import pandas as pd
import streamlit as st

FEATURES = [
    "study_hours_per_week",
    "attendance_percentage",
    "previous_exam_score",
    "sleep_hours_per_night",
    "revision_frequency_per_week",
]

LABELS = {
    "study_hours_per_week": "Study Hours / Week",
    "attendance_percentage": "Attendance %",
    "previous_exam_score": "Previous Exam Score",
    "sleep_hours_per_night": "Sleep Hours / Night",
    "revision_frequency_per_week": "Revision Days / Week",
}

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Student Academic Performance Predictor")
st.write("### CBSE Class 11 AI Capstone Project")
st.write(
    "Enter the student's details below to predict the final academic score."
)

# Load trained model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("model.pkl was not found in the project files.")
    st.stop()

# Student inputs
st.sidebar.header("Student Details")

study = st.sidebar.slider(
    "Study Hours / Week",
    1.0, 40.0, 15.0, 0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    50.0, 100.0, 80.0, 0.5
)

previous = st.sidebar.slider(
    "Previous Exam Score",
    0.0, 100.0, 60.0, 0.5
)

sleep = st.sidebar.slider(
    "Sleep Hours / Night",
    4.0, 10.0, 7.0, 0.5
)

revision = st.sidebar.slider(
    "Revision Days / Week",
    0, 7, 3
)

# Prepare input
inputs = pd.DataFrame(
    [[study, attendance, previous, sleep, revision]],
    columns=FEATURES
)

# Prediction
if st.button("🎯 Predict Final Score", use_container_width=True):

    predicted = float(model.predict(inputs)[0])

    predicted = max(0.0, min(100.0, predicted))

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Final Score",
        f"{predicted:.1f} / 100"
    )

    col2.metric(
        "Change vs Previous Exam",
        f"{predicted - previous:+.1f}"
    )

    st.progress(int(predicted))

    if predicted < 33:
        st.error("🔴 At Risk")
    elif predicted < 45:
        st.warning("🟡 Borderline")
    elif predicted < 75:
        st.info("🔵 On Track")
    else:
        st.success("🟢 Excellent")

    # Suggestions
    suggestions = []

    if attendance < 75:
        suggestions.append(
            "Improve attendance."
        )

    if study < 10:
        suggestions.append(
            "Gradually increase study time."
        )

    if revision < 3:
        suggestions.append(
            "Revise at least 3-4 days per week."
        )

    if sleep < 6 or sleep > 9:
        suggestions.append(
            "Aim for around 7-8 hours of sleep."
        )

    st.subheader("💡 Suggestions")

    if suggestions:
        for suggestion in suggestions:
            st.write("• " + suggestion)
    else:
        st.write(
            "✅ Keep maintaining your current study routine."
        )

    # Feature importance
    st.subheader("📊 Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    )

    importance.index = [
        LABELS[f] for f in importance.index
    ]

    st.bar_chart(
        importance.sort_values(ascending=False)
    )

    st.caption(
        "Higher values indicate greater relative influence "
        "on the model's predictions."
    )
