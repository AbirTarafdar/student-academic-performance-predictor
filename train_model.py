"""Modelling & Evaluation: train a RandomForestRegressor and save it."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate on unseen data
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root MSE (average error in marks): {np.sqrt(mse):.2f}")
print(f"R2 Score: {r2:.3f}  (1.0 = perfect, 0 = no better than guessing the average)")

print("\nFeature importance:")
for name, score in sorted(zip(FEATURES, model.feature_importances_),
                          key=lambda t: -t[1]):
    print(f"  {name:32s} {score:.3f}")

# 5. Save
joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")