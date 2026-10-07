import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME-FRIENDLY STYLING
# IMPORTANT: No custom HTML cards are used for the results.
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero-box {
        padding: 1.5rem 1rem;
        border-radius: 18px;
        text-align: center;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--text-color);
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
    }

    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.55;
        padding-top: 1.5rem;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 600;
        font-size: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-box">
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


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load("model.pkl")

except FileNotFoundError:
    st.error(
        "model.pkl was not found in the project files."
    )
    st.stop()

except Exception as error:
    st.error(
        "The trained model could not be loaded."
    )
    st.caption(f"Technical detail: {error}")
    st.stop()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

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


# ============================================================
# INPUT DATA
# ============================================================

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


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_clicked = st.button(
    "🎯 Predict Final Score",
    use_container_width=True,
)


if predict_clicked:

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:
        predicted = float(model.predict(inputs)[0])

    except Exception as error:
        st.error("The model could not generate a prediction.")
        st.caption(f"Technical detail: {error}")
        st.stop()

    # Keep score within valid 0-100 range.
    predicted = max(0.0, min(100.0, predicted))


    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Prediction Result")

    st.caption(
        "The AI model has analyzed the information provided."
    )


    # --------------------------------------------------------
    # MAIN RESULTS
    # --------------------------------------------------------

    change = predicted - previous

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Final Score",
            value=f"{predicted:.1f} / 100",
        )

    with col2:

        if change > 0:
            change_text = f"+{change:.1f}"
        else:
            change_text = f"{change:.1f}"

        st.metric(
            label="Change vs Previous Exam",
            value=change_text,
            delta=f"{change:+.1f} marks",
        )


    # --------------------------------------------------------
    # EXPLANATION OF CHANGE
    # --------------------------------------------------------

    if change > 0:

        st.success(
            f"📈 The predicted score is {change:.1f} marks "
            "higher than the previous exam."
        )

    elif change < 0:

        st.warning(
            f"📉 The model predicts a score "
            f"{abs(change):.1f} marks lower than the previous exam."
        )

    else:

        st.info(
            "➡️ The predicted score is the same as the previous exam."
        )


    # --------------------------------------------------------
    # SCORE PROGRESS
    # --------------------------------------------------------

    st.write("**Overall Predicted Score**")

    st.progress(
        int(round(predicted)),
        text=f"{predicted:.1f} / 100",
    )


    # --------------------------------------------------------
    # PERFORMANCE STATUS
    # --------------------------------------------------------

    if predicted < 33:

        st.error(
            "🔴 **At Risk**\n\n"
            "The predicted score indicates that additional "
            "academic support may be helpful."
        )

    elif predicted < 45:

        st.warning(
            "🟡 **Borderline**\n\n"
            "The student is close to the passing range. "
            "Consistent improvement can help."
        )

    elif predicted < 75:

        st.info(
            "🔵 **On Track**\n\n"
            "The student is currently on a positive academic track."
        )

    else:

        st.success(
            "🟢 **Excellent**\n\n"
            "The predicted performance is in an excellent range."
        )


    # --------------------------------------------------------
    # PERSONALIZED SUGGESTIONS
    # --------------------------------------------------------

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

    if sleep > 9:
        suggestions.append(
            "Maintain a consistent sleep routine."
        )

    if suggestions:

        for suggestion in suggestions:
            st.info("• " + suggestion)

    else:

        st.success(
            "✅ Keep maintaining your current study routine."
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("📊 AI Feature Importance")

    if hasattr(model, "feature_importances_"):

        raw_importance = model.feature_importances_

        if len(raw_importance) == len(FEATURES):

            importance = pd.Series(
                raw_importance,
                index=FEATURES,
                dtype="float64",
            )

            # Remove tiny negative numerical noise if present.
            importance = importance.clip(lower=0)

            total = float(importance.sum())

            if total > 0:

                # Convert raw importance to percentages.
                percentage = (
                    importance / total
                ) * 100

                chart_data = pd.DataFrame(
                    {
                        "Feature": [
                            LABELS[f]
                            for f in FEATURES
                        ],
                        "Importance": importance.values,
                        "Percentage": percentage.values,
                    }
                )

                # Text shown directly on each slice.
                chart_data["Label"] = chart_data[
                    "Percentage"
                ].apply(
                    lambda x: f"{x:.1f}%"
                )


                # ------------------------------------------------
                # PIE CHART
                # ------------------------------------------------

                pie_chart = {

                    "width": "container",

                    "height": 450,

                    "layer": [

                        # Pie slices
                        {
                            "mark": {
                                "type": "arc",
                                "outerRadius": 155,
                                "stroke": "white",
                                "strokeWidth": 1,
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
                                ],
                            },
                        },

                        # Percentage labels
                        {
                            "mark": {
                                "type": "text",
                                "radius": 105,
                                "fontSize": 16,
                                "fontWeight": "normal",
                            },

                            "encoding": {

                                "theta": {
                                    "field": "Importance",
                                    "type": "quantitative",
                                    "stack": "normalize",
                                },

                                "text": {
                                    "field": "Label",
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
                }


                st.vega_lite_chart(
                    chart_data,
                    pie_chart,
                    use_container_width=True,
                )

                st.caption(
                    "Percentages show each feature's relative "
                    "importance in the trained AI model. "
                    "They do not represent the student's "
                    "percentage of effort."
                )

            else:

                st.info(
                    "The model returned zero feature-importance values."
                )

        else:

            st.info(
                "The model's feature-importance data does not "
                "match the five input features."
            )

    else:

        st.info(
            "This trained model does not provide feature importance."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        🎓 CBSE Class 11 AI Capstone Project<br>
        Student Academic Performance Predictor
    </div>
    """,
    unsafe_allow_html=True,
)
