"""Modelling & Evaluation: train a RandomForestRegressor and save it."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "study_hours_per_week",
    "attendance_percentage",
    "previous_exam_score",
    "sleep_hours_per_night",
    "revision_frequency_per_week",
]
TARGET = "final_score"

# 1. Load data
df = pd.read_csv("student_data.csv")
X, y = df[FEATURES], df[TARGET]

# 2. Split: 80% to learn from, 20% kept hidden to test the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train
model = RandomForestRegressor(
    n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate on training data and on unseen data
for label, (a, b) in {"Training": (X_train, y_train), "Testing": (X_test, y_test)}.items():
    p = model.predict(a)
    print(f"{label:9s} R2={r2_score(b, p):.3f}  "
          f"MAE={mean_absolute_error(b, p):.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(b, p)):.2f}")
print("(R2: 1.0 = perfect, 0 = no better than guessing the average)")

print("\nFeature importance:")
for name, score in sorted(zip(FEATURES, model.feature_importances_),
                          key=lambda t: -t[1]):
    print(f"  {name:32s} {score:.3f}")

# 5. Save
joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")

# 6. Quick sanity check with the strong-student example
demo = pd.DataFrame([[36, 95, 92, 9.5, 7]], columns=FEATURES)
print(f"\nSanity check (36, 95, 92, 9.5, 7) -> {model.predict(demo)[0]:.1f}")
