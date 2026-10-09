
# ==========================================
# MULTIPLE LINEAR REGRESSION
# Hotel Booking Demand
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# 1. SETUP
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "hotel_bookings_cleaned.csv"
FIGURE_DIR = BASE_DIR / "results" / "figures" / "regression"
TABLE_DIR = BASE_DIR / "results" / "tables" / "regression"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# ==========================================
# 3. SELECT VARIABLES
# ==========================================
target = "adr"
features = [
    "lead_time",
    "adults",
    "children",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "booking_changes",
    "total_of_special_requests"
]

data = df[features + [target]].copy()

# Convert selected columns to numeric
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove missing and infinite values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
X = data[features]
y = data[target]
print("\nData used for regression:", data.shape)
print("\nFeatures:")
print(features)

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. BUILD MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# ==========================================
# 6. MODEL EVALUATION
# ==========================================
r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n========== MODEL PERFORMANCE ==========")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adjusted_r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ==========================================
# 7. REGRESSION COEFFICIENTS
# ==========================================
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n========== REGRESSION COEFFICIENTS ==========")
print("Intercept:", model.intercept_)
print(coefficients)

# ==========================================
# 8. SAVE RESULTS
# ==========================================
metrics = pd.DataFrame({
    "Metric": [
        "R-squared",
        "Adjusted R-squared",
        "MAE",
        "RMSE"
    ],
    "Value": [
        r2,
        adjusted_r2,
        mae,
        rmse
    ]
})

metrics.to_csv(TABLE_DIR / "regression_metrics.csv", index=False)
coefficients.to_csv(TABLE_DIR / "regression_coefficients.csv", index=False)

predictions = pd.DataFrame({
    "Actual_ADR": y_test.values,
    "Predicted_ADR": y_pred
})

predictions.to_csv(TABLE_DIR / "regression_predictions.csv", index=False)

# ==========================================
# 9. ACTUAL VS PREDICTED
# ==========================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual ADR")
plt.ylabel("Predicted ADR")
plt.title("Actual vs Predicted ADR")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# ==========================================
# 10. RESIDUAL PLOT
# ==========================================
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(
    y_pred,
    residuals,
    alpha=0.3
)
plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted ADR")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "residual_plot.png",
    dpi=300
)

plt.close()
print("\nRegression analysis completed!")