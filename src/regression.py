# ==========================================
# MULTIPLE LINEAR REGRESSION
# Hotel Booking Demand
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "hotel_bookings_cleaned_pre_capping.csv"
)

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

# Biến mục tiêu
target = "adr"

# Các biến đầu vào dạng số
numeric_features = [
    "lead_time",
    "adults",
    "children",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "booking_changes",
    "total_of_special_requests"
]

# Các biến đầu vào dạng phân loại
categorical_features = [
    "hotel",
    "arrival_date_month",
    "market_segment",
    "customer_type",
    "meal",
    "is_repeated_guest"
]

features = numeric_features + categorical_features

# Kiểm tra các cột có tồn tại trong dataset
required_columns = features + [target]
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset thiếu các cột: {missing_columns}"
    )

# Chỉ lấy các cột cần thiết
data = df[required_columns].copy()

# Chuyển các biến số sang kiểu numeric
for col in numeric_features + [target]:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# Thay giá trị vô hạn bằng NaN
data = data.replace([np.inf, -np.inf], np.nan)

# Loại bỏ dòng thiếu biến mục tiêu
data = data.dropna(subset=[target])

# Các giá trị thiếu ở những biến này sẽ được xử lý
# bằng SimpleImputer trong Pipeline.

X = data[features].copy()
y = data[target].copy()

print("\nData used for regression:", data.shape)

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tạo bản sao để tránh cảnh báo khi chỉnh sửa DataFrame
X_train = X_train.copy()
X_test = X_test.copy()

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 5. CAPPING NUMERIC FEATURES
# ==========================================

# Chỉ capping một số biến số đầu vào.
# Không capping ADR vì đây là biến mục tiêu.
cap_columns = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights"
]

# Tính ngưỡng chỉ từ tập train
cap_bounds = {}

for col in cap_columns:
    q1 = X_train[col].quantile(0.25)
    q3 = X_train[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr

    cap_bounds[col] = (lower, upper)

    # Áp dụng cùng ngưỡng lên train và test
    X_train.loc[:, col] = X_train[col].clip(
        lower=lower,
        upper=upper
    )

    X_test.loc[:, col] = X_test[col].clip(
        lower=lower,
        upper=upper
    )

    print(
        f"{col}: lower={lower:.4f}, "
        f"upper={upper:.4f}"
    )

# ==========================================
# 6. BUILD PREPROCESSING PIPELINE
# ==========================================

# Xử lý biến số: điền giá trị thiếu bằng median
numeric_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median"))
])

# Xử lý biến phân loại:
# 1. Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất
# 2. Chuyển biến chữ thành các cột nhị phân
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Ghép quy trình xử lý hai nhóm biến
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==========================================
# 7. BUILD AND TRAIN MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# ==========================================
# 8. MODEL EVALUATION
# ==========================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Số đặc trưng sau khi One-Hot Encoding
feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

n = len(y_test)
p = len(feature_names)

print("\n========== MODEL PERFORMANCE ==========")
print(f"R-squared: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Number of encoded features: {p}")

# ==========================================
# 9. REGRESSION COEFFICIENTS
# ==========================================

# Lấy hệ số của mô hình Linear Regression
regressor = model.named_steps["regressor"]
coefficients = regressor.coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sắp xếp theo trị tuyệt đối để xem hệ số lớn trước
coef_df["Abs_Coefficient"] = (
    coef_df["Coefficient"].abs()
)

coef_df = coef_df.sort_values(
    by="Abs_Coefficient",
    ascending=False
)

print("\n========== REGRESSION COEFFICIENTS ==========")
print(f"Intercept: {regressor.intercept_:.6f}")
print(
    coef_df.drop(
        columns=["Abs_Coefficient"]
    ).to_string(index=False)
)

# ==========================================
# 10. SAVE RESULTS
# ==========================================

metrics = pd.DataFrame({
    "Metric": [
        "R-squared",
        "MAE",
        "RMSE",
        "Number of encoded features"
    ],
    "Value": [
        r2,
        mae,
        rmse,
        p
    ]
})

metrics.to_csv(
    TABLE_DIR / "regression_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

coef_df.drop(
    columns=["Abs_Coefficient"]
).to_csv(
    TABLE_DIR / "regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions = pd.DataFrame({
    "Actual_ADR": y_test.to_numpy(),
    "Predicted_ADR": y_pred
})

predictions.to_csv(
    TABLE_DIR / "regression_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

# Lưu ngưỡng capping để tiện trình bày trong báo cáo
cap_summary = pd.DataFrame([
    {
        "Feature": col,
        "Lower_Bound": bounds[0],
        "Upper_Bound": bounds[1]
    }
    for col, bounds in cap_bounds.items()
])

cap_summary.to_csv(
    TABLE_DIR / "regression_capping_bounds.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 11. ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.3
)

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
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 12. RESIDUAL PLOT
# ==========================================

residuals = y_test.to_numpy() - y_pred

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
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRegression analysis completed!")
print("Results saved to:", TABLE_DIR)
print("Figures saved to:", FIGURE_DIR)
