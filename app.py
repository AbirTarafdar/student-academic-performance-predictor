import joblib
import pandas as pd
import streamlit as st

# ==================================================
# CONFIGURATION
# ==================================================

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
    initial_sidebar_state="expanded",
)

# ==================================================
# THEME-AWARE CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    /* ---------- GENERAL ---------- */

    .main {
        padding-top: 1rem;
    }

    /* ---------- HERO ---------- */

    .hero {
        padding: 1.6rem 1.2rem;
        border-radius: 20px;
        text-align: center;

        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);

        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.5rem;

        color: var(--text-color);
    }

    .hero-subtitle {
        font-size: 1rem;

        color: var(--text-color);
        opacity: 0.70;

        margin-bottom: 0;
    }

    /* ---------- SECTION CARDS ---------- */

    .section-card {
        padding: 1.2rem;
        border-radius: 16px;

        background: var(--secondary-background-color);
        color: var(--text-color);

        border: 1px solid rgba(128, 128, 128, 0.25);

        margin-bottom: 1rem;
    }

    .section-card h3 {
        color: var(--text-color);
        margin-bottom: 0.4rem;
    }

    .section-card p {
        color: var(--text-color);
        opacity: 0.75;
    }

    /* ---------- RESULT CARDS ---------- */

    .result-card {
        padding: 1.25rem 0.8rem;
        border-radius: 16px;
        text-align: center;

        background: var(--secondary-background-color);
        color: var(--text-color);

        border: 1px solid rgba(128, 128, 128, 0.25);

        min-height: 105px;

        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .result-number {
        font-size: 2rem;
        font-weight: 750;
        color: var(--text-color);
        line-height: 1.2;
    }

    .result-label {
        font-size: 0.9rem;

        color: var(--text-color);
        opacity: 0.70;

        margin-bottom: 0.35rem;
    }

    /* ---------- STATUS ---------- */

    .status-card {
        padding: 1rem;
        border-radius: 14px;

        text-align: center;

        margin: 1rem 0;

        font-weight: 600;

        border: 1px solid rgba(128, 128, 128, 0.20);
    }

    /* ---------- INFO BOX ---------- */

    .info-card {
        padding: 0.9rem 1rem;
        border-radius: 12px;

        background: var(--secondary-background-color);
        color: var(--text-color);

        border: 1px solid rgba(128, 128, 128, 0.22);

        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;

        color: var(--text-color);
        opacity: 0.55;

        font-size: 0.8rem;

        padding: 1.5rem 0 0.5rem 0;
    }

    /* ---------- BUTTON ---------- */

    div.stButton > button {
        border-radius: 12px;
        min-height: 3rem;

        font-weight: 650;
        font-size: 1rem;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);

        border: 1px solid rgba(128, 128, 128, 0.25);

        padding: 0.8rem;

        border-radius: 14px;

        color: var(--text-color);
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

    /* ---------- DIVIDERS ---------- */

    hr {
        border-color: rgba(128, 128, 128, 0.20);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🎓 Student Academic Performance Predictor
        </div>

        <div class="hero-subtitle">
            CBSE Class 11 AI Capstone Project
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter the student's academic and lifestyle details "
    "to estimate their final academic score."
)

# ==================================================
# LOAD MODEL
# ==================================================

try:
    model = joblib.load("model.pkl")

except FileNotFoundError:
    st.error(
        "❌ model.pkl was not found in the project files. "
        "Please make sure model.pkl is uploaded with this app."
    )
    st.stop()

except Exception as error:
    st.error(
        "❌ The trained model could not be loaded."
    )
    st.caption(f"Technical detail: {error}")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🎯 Student Details")

st.sidebar.caption(
    "Adjust the values below and run the prediction."
)

study = st.sidebar.slider(
    "📚 Study Hours / Week",
    min_value=1.0,
    max_value=40.0,
    value=15.0,
    step=0.5,
)

attendance = st.sidebar.slider(
    "🏫 Attendance (%)",
    min_value=50.0,
    max_value=100.0,
    value=80.0,
    step=0.5,
)

previous = st.sidebar.slider(
    "📝 Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.5,
)

sleep = st.sidebar.slider(
    "😴 Sleep Hours / Night",
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
)

revision = st.sidebar.slider(
    "🔄 Revision Days / Week",
    min_value=0,
    max_value=7,
    value=3,
    step=1,
)

# ==================================================
# INPUT DATA
# ==================================================

inputs = pd.DataFrame(
    [[
        study,
        attendance,
        previous,
        sleep,
        revision,
    ]],
    columns=FEATURES,
)

# ==================================================
# PREDICTION
# ==================================================

if st.button(
    "🎯 Predict Final Score",
    use_container_width=True,
):

    try:
        predicted = float(model.predict(inputs)[0])

    except Exception as error:
        st.error(
            "❌ The model could not generate a prediction."
        )
        st.caption(f"Technical detail: {error}")
        st.stop()

    # Keep prediction within valid score range.
    predicted = max(0.0, min(100.0, predicted))

    # ==================================================
    # RESULT HEADER
    # ==================================================

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

    # ==================================================
    # RESULT METRICS
    # ==================================================

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

        # IMPORTANT:
        # This is intentionally NOT forced positive.
        # It represents the actual difference between
        # the model prediction and the previous score.

        change = predicted - previous

        if change > 0:
            change_display = f"+{change:.1f}"
        elif change < 0:
            change_display = f"{change:.1f}"
        else:
            change_display = "0.0"

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Change vs Previous Exam
                </div>

                <div class="result-number">
                    {change_display}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==================================================
    # SCORE INTERPRETATION
    # ==================================================

    if change > 0:
        st.success(
            f"📈 The predicted score is {change:.1f} marks "
            f"higher than the previous exam."
        )

    elif change < 0:
        st.warning(
            f"📉 The model predicts a score {abs(change):.1f} marks "
            f"lower than the previous exam."
        )

    else:
        st.info(
            "➡️ The predicted score is the same as the previous exam."
        )

    # ==================================================
    # SCORE PROGRESS
    # ==================================================

    st.write("**Overall Predicted Score**")

    st.progress(
        int(round(predicted)),
        text=f"{predicted:.1f} / 100",
    )

    # ==================================================
    # PERFORMANCE STATUS
    # ==================================================

    if predicted < 33:

        status = "🔴 At Risk"
        status_message = (
            "The predicted score indicates that additional "
            "academic support may be helpful."
        )
        status_color = "rgba(220, 60, 60, 0.15)"

    elif predicted < 45:

        status = "🟡 Borderline"
        status_message = (
            "The student is close to the passing range. "
            "Consistent improvement can help."
        )
        status_color = "rgba(240, 190, 40, 0.15)"

    elif predicted < 75:

        status = "🔵 On Track"
        status_message = (
            "The student is currently on a positive "
            "academic track."
        )
        status_color = "rgba(60, 130, 230, 0.15)"

    else:

        status = "🟢 Excellent"
        status_message = (
            "The predicted performance is in an "
            "excellent range."
        )
        status_color = "rgba(60, 180, 90, 0.15)"

    st.markdown(
        f"""
        <div
            class="status-card"
            style="background:{status_color};"
        >

            <div style="font-size:1.2rem;">
                {status}
            </div>

            <div
                style="
                    font-size:0.9rem;
                    font-weight:400;
                    margin-top:5px;
                "
            >
                {status_message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================================================
    # SUGGESTIONS
    # ==================================================

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
            "Revise at least 3–4 days per week."
        )

    if sleep < 6:
        suggestions.append(
            "Try to maintain a healthier sleep routine."
        )

    elif sleep > 9:
        suggestions.append(
            "Your sleep duration is relatively high; "
            "keep a consistent routine."
        )

    if suggestions:

        for suggestion in suggestions:
            st.info("• " + suggestion)

    else:

        st.success(
            "✅ Keep maintaining your current study routine."
        )

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    st.subheader("📊 AI Feature Importance")

    # Check that the model supports feature importance.
    if hasattr(model, "feature_importances_"):

        raw_importance = model.feature_importances_

        importance = pd.Series(
            raw_importance,
            index=FEATURES,
            dtype="float64",
        )

        # Remove tiny numerical noise.
        importance = importance.clip(lower=0)

        total_importance = importance.sum()

        if total_importance > 0:

            # Convert model importance into percentages.
            importance_percentage = (
                importance / total_importance * 100
            )

            chart_data = pd.DataFrame(
                {
                    "Feature": [
                        LABELS[f] for f in FEATURES
                    ],
                    "Importance": importance.values,
                    "Percentage": importance_percentage.values,
                }
            )

            # Round only for display.
            chart_data["Percentage Label"] = (
                chart_data["Percentage"]
                .map(lambda value: f"{value:.1f}%")
            )

            # --------------------------------------------------
            # PIE CHART WITH PERCENTAGES
            # --------------------------------------------------

            st.vega_lite_chart(
                chart_data,
                {
                    "width": "container",
                    "height": 420,

                    "layer": [

                        # Pie slices
                        {
                            "mark": {
                                "type": "arc",
                                "outerRadius": 150,
                                "stroke": "white",
                                "strokeWidth": 2,
                            },

                            "encoding": {
                                "theta": {
                                    "field": "Importance",
                                    "type": "quantitative",
                                },

                                "color": {
                                    "field": "Feature",
                                    "type": "nominal",
                                    "legend": {
                                        "title": "Features",
                                        "orient": "right",
                                    },
                                },

                                "tooltip": [
                                    {
                                        "field": "Feature",
                                        "type": "nominal",
                                        "title": "Feature",
                                    },
                                    {
                                        "field": "Percentage",
                                        "type": "quantitative",
                                        "format": ".1f",
                                        "title": "Importance (%)",
                                    },
                                    {
                                        "field": "Importance",
                                        "type": "quantitative",
                                        "format": ".3f",
                                        "title": "Raw Importance",
                                    },
                                ],
                            },
                        },

                        # Percentage labels
                        {
                            "mark": {
                                "type": "text",
                                "radius": 105,
                                "fontSize": 13,
                                "fontWeight": "bold",
                                "fill": "white",
                                "stroke": "#333333",
                                "strokeWidth": 2,
                            },

                            "encoding": {
                                "theta": {
                                    "field": "Importance",
                                    "type": "quantitative",
                                    "stack": "normalize",
                                },

                                "text": {
                                    "field": "Percentage Label",
                                    "type": "nominal",
                                },

                                "color": {
                                    "value": "white",
                                },

                                "detail": {
                                    "field": "Feature",
                                },
                            },
                        },
                    ],

                    "view": {
                        "stroke": None,
                    },
                },
                use_container_width=True,
            )

            st.caption(
                "The percentages represent the model's relative "
                "feature importance. They do not represent the "
                "student's percentage of effort."
            )

        else:

            st.info(
                "Feature importance values are unavailable "
                "for this prediction."
            )

    else:

        st.info(
            "This trained model does not provide feature "
            "importance values."
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        🎓 CBSE Class 11 AI Capstone Project<br>
        Student Academic Performance Predictor
    </div>
    """,
    unsafe_allow_html=True,
)

This version keeps your current model calculation honest, adds the percentage labels, and makes the custom cards/theme styling adapt much better between light and dark mode.

After replacing "app.py": save/commit it, wait for Streamlit Cloud to show the new deployment, then test the same "36 / 95 / 92 / 9.5 / 7" inputs. If it still gives 84.9, that's confirmation that the remaining issue is inside "model.pkl", not this interface code.
