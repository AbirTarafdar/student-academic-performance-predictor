"""Data Acquisition: generate 1000 realistic synthetic student records."""
import numpy as np
import pandas as pd

N_RECORDS = 1000
rng = np.random.default_rng(seed=42)


def generate_students(n: int) -> pd.DataFrame:
    study = np.clip(rng.normal(18, 9, n), 1.0, 40.0)
    attendance = np.clip(rng.normal(82, 11, n), 50.0, 100.0)
    sleep = np.clip(rng.normal(7, 1.2, n), 4.0, 10.0)
    revision = rng.integers(0, 8, n)

    # Previous score: loosely linked to study and attendance, with a wide
    # spread so the data covers the full 0-100 range used by the app sliders.
    previous = np.clip(
        20 + 0.6 * study
        + 0.2 * (attendance - 50)
        + rng.normal(0, 18, n)
        + rng.uniform(0, 25, n),
        0.0,
        100.0,
    )

    final = (
        0.45 * previous
        + 0.9 * study
        + 0.18 * attendance
        + 1.2 * revision
        - 1.5 * np.abs(sleep - 7.5)
        + 5
        + rng.normal(0, 4, n)
    )
    final = np.clip(final, 0.0, 100.0)

    return pd.DataFrame({
        "study_hours_per_week": study.round(1),
        "attendance_percentage": attendance.round(1),
        "previous_exam_score": previous.round(1),
        "sleep_hours_per_night": sleep.round(1),
        "revision_frequency_per_week": revision,
        "final_score": final.round(1),
    })


if __name__ == "__main__":
    df = generate_students(N_RECORDS)
    df.to_csv("student_data.csv", index=False)

    print(f"Saved {len(df)} records to student_data.csv")
    print(df.describe().round(2))
