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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        background: linear-gradient(135deg, #eef4ff, #f7f9fc);
        border: 1px solid #d9e2f2;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #5f6b7a;
        margin-bottom: 0;
    }

    /* Section cards */
    .section-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e1e6ef;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        background: #f8fbff;
        border: 1px solid #d8e7ff;
    }

    .result-number {
        font-size: 2rem;
        font-weight: 700;
    }

    .result-label {
        font-size: 0.9rem;
        color: #657080;
    }

    /* Status */
    .status-card {
        padding: 0.9rem;
        border-radius: 14px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7a8491;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    /* Button */
    div.stButton > button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: #f8fbff;
        border: 1px solid #e1e8f2;
        padding: 0.8rem;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎓 Student Academic Performance Predictor</div>
        <div class="hero-subtitle">
            CBSE Class 11 AI Capstone Project
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter the student's academic and lifestyle details to estimate "
    "their final academic score."
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("model.pkl was not found in the project files.")
    st.stop()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.title("🎯 Student Details")
st.sidebar.caption("Adjust the values below and run the prediction.")

study = st.sidebar.slider(
    "📚 Study Hours / Week",
    1.0,
    40.0,
    15.0,
    0.5,
)

attendance = st.sidebar.slider(
    "🏫 Attendance (%)",
    50.0,
    100.0,
    80.0,
    0.5,
)

previous = st.sidebar.slider(
    "📝 Previous Exam Score",
    0.0,
    100.0,
    60.0,
    0.5,
)

sleep = st.sidebar.slider(
    "😴 Sleep Hours / Night",
    4.0,
    10.0,
    7.0,
    0.5,
)

revision = st.sidebar.slider(
    "🔄 Revision Days / Week",
    0,
    7,
    3,
)

# --------------------------------------------------
# INPUT DATA
# --------------------------------------------------

inputs = pd.DataFrame(
    [[study, attendance, previous, sleep, revision]],
    columns=FEATURES,
)

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🎯 Predict Final Score",
    use_container_width=True,
):

    # Prediction
    predicted = float(model.predict(inputs)[0])

    predicted = max(0.0, min(100.0, predicted))

    # --------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------

    st.markdown(
        """
        <div class="section-card">
            <h3>📊 Prediction Result</h3>
            <p>
                The AI model has analyzed the information provided.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Predicted Final Score
                </div>
                <div class="result-number">
                    {predicted:.1f} / 100
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        change = predicted - previous
        sign = "+" if change >= 0 else ""

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Change vs Previous Exam
                </div>
                <div class="result-number">
                    {sign}{change:.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------
    # SCORE PROGRESS
    # --------------------------------------------------

    st.write("**Overall Score Progress**")
    st.progress(int(predicted))

    # --------------------------------------------------
    # PERFORMANCE STATUS
    # --------------------------------------------------

    if predicted < 33:
        status = "🔴 At Risk"
        status_message = "The predicted score indicates that additional academic support may be helpful."
        status_color = "#ffe5e5"

    elif predicted < 45:
        status = "🟡 Borderline"
        status_message = "The student is close to the passing range. Consistent improvement can help."

        status_color = "#fff4cc"

    elif predicted < 75:
        status = "🔵 On Track"
        status_message = "The student is currently on a positive academic track."

        status_color = "#e5f0ff"

    else:
        status = "🟢 Excellent"
        status_message = "The predicted performance is in an excellent range."

        status_color = "#e5f7e8"

    st.markdown(
        f"""
        <div class="status-card" style="background:{status_color};">
            <div style="font-size:1.2rem;">{status}</div>
            <div style="font-size:0.9rem; font-weight:400; margin-top:5px;">
                {status_message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # SUGGESTIONS
    # --------------------------------------------------

    st.subheader("💡 Personalized Suggestions")

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

    if suggestions:
        for suggestion in suggestions:
            st.info("• " + suggestion)
    else:
        st.success(
            "✅ Keep maintaining your current study routine."
        )

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    st.subheader("📊 AI Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    )

    importance.index = [
        LABELS[f] for f in importance.index
    ]

    chart_data = pd.DataFrame(
        {
            "Feature": importance.index,
            "Importance": importance.values,
        }
    )

    st.vega_lite_chart(
        chart_data,
        {
            "mark": "arc",
            "encoding": {
                "theta": {
                    "field": "Importance",
                    "type": "quantitative",
                },
                "color": {
                    "field": "Feature",
                    "type": "nominal",
                },
                "tooltip": [
                    {
                        "field": "Feature",
                        "type": "nominal",
                        "title": "Feature",
                    },
                    {
                        "field": "Importance",
                        "type": "quantitative",
                        "format": ".3f",
                        "title": "Importance",
                    },
                ],
            },
            "view": {
                "stroke": None,
            },
        },
        use_container_width=True,
    )

    st.caption(
        "Higher values indicate greater relative influence "
        "on the model's predictions."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        🎓 CBSE Class 11 AI Capstone Project<br>
        Student Academic Performance Predictor
    </div>
    """,
    unsafe_allow_html=True,
)
