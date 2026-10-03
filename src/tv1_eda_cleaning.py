"""
TV1 - EDA & Data Cleaning
Dataset: Hotel Booking Demand

Cấu trúc project:

Hotel-Booking-Demand/
│
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv
│   │
│   └── processed/
│       └── các file sau khi xử lý
│
├── reports/
│   └── figures/
│       └── các biểu đồ
│
└── src/
    └── tv1_eda_cleaning.py

Chạy file từ thư mục src hoặc từ thư mục gốc đều được.
"""

# ============================================================
# 1. IMPORT THƯ VIỆN
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "hotel_bookings.csv"

# Lưu dữ liệu đã xử lý
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Lưu tất cả biểu đồ vào results/figures
RESULTS_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

# Tự động tạo thư mục nếu chưa có
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("ĐƯỜNG DẪN LƯU FILE")
print("=" * 80)

print(f"BASE_DIR:")
print(BASE_DIR)

print(f"\nPROCESSED_DIR:")
print(PROCESSED_DIR)

print(f"\nRESULTS_DIR:")
print(RESULTS_DIR)

print(f"\nFIGURE_DIR:")
print(FIGURE_DIR)

# ============================================================
# 3. ĐỌC DỮ LIỆU
# ============================================================

print("\n" + "=" * 80)
print("[1] ĐỌC DỮ LIỆU")
print("=" * 80)

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file dữ liệu:\n{RAW_PATH}\n"
        "Hãy kiểm tra lại thư mục data/raw."
    )

df = pd.read_csv(RAW_PATH)

print(f"Kích thước dữ liệu: {df.shape}")
print(f"Số dòng: {df.shape[0]:,}")
print(f"Số cột: {df.shape[1]}")


# ============================================================
# 4. XEM THÔNG TIN CƠ BẢN
# ============================================================

print("\n" + "=" * 80)
print("[2] THÔNG TIN CƠ BẢN")
print("=" * 80)

print("\n5 dòng đầu tiên:")
print(df.head())

print("\nTên các cột:")
print(df.columns.tolist())

print("\nThông tin dữ liệu:")
df.info()


# ============================================================
# 5. KIỂM TRA CÁC CỘT CẦN THIẾT
# ============================================================

print("\n" + "=" * 80)
print("[3] KIỂM TRA CỘT CẦN THIẾT")
print("=" * 80)

required_columns = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "adults",
    "children",
    "babies",
    "country",
    "market_segment",
    "deposit_type",
    "adr",
    "company",
    "agent",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "days_in_waiting_list",
]

missing_required_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required_columns:
    raise ValueError(
        "Thiếu các cột cần thiết:\n"
        + "\n".join(missing_required_columns)
    )

print("Tất cả các cột cần thiết đều tồn tại.")


# ============================================================
# 6. KIỂM TRA DUPLICATE
# ============================================================

print("\n" + "=" * 80)
print("[4] KIỂM TRA DỮ LIỆU TRÙNG LẶP")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(f"Số dòng bị trùng hoàn toàn: {duplicate_count:,}")

if duplicate_count > 0:
    print("Có dữ liệu trùng lặp và sẽ được loại bỏ ở bước làm sạch.")
else:
    print("Không có dòng trùng hoàn toàn.")


# ============================================================
# 7. KIỂM TRA GIÁ TRỊ THIẾU
# ============================================================

print("\n" + "=" * 80)
print("[5] KIỂM TRA GIÁ TRỊ THIẾU")
print("=" * 80)

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0].sort_values(
    ascending=False
)

print("\nGiá trị thiếu trước khi xử lý:")

if len(missing_before) > 0:
    print(missing_before)
else:
    print("Không có giá trị thiếu.")


# Lưu thống kê missing trước khi xử lý.
missing_before_df = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_before_df.columns = [
    "column",
    "missing_count"
]

missing_before_df["missing_percent"] = (
    missing_before_df["missing_count"]
    / len(df)
    * 100
)

missing_before_df = missing_before_df[
    missing_before_df["missing_count"] > 0
]

missing_before_path = (
    PROCESSED_DIR / "missing_values_before.csv"
)

missing_before_df.to_csv(
    missing_before_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. TẠO CÁC CỘT ĐÁNH DẤU MISSING
# ============================================================

# Lưu lại thông tin một số cột bị thiếu trước khi thay thế.
# Việc này giúp vẫn biết được bản ghi ban đầu có bị thiếu hay không.

df["agent_was_missing"] = df["agent"].isna()
df["children_was_missing"] = df["children"].isna()
df["country_was_missing"] = df["country"].isna()

# company có rất nhiều giá trị thiếu.
# Thay vì điền giả một giá trị cho company,
# ta tạo biến has_company để biểu diễn việc booking có company hay không.
df["has_company"] = df["company"].notna().astype(int)


# ============================================================
# 9. XỬ LÝ CỘT COMPANY
# ============================================================

print("\n" + "=" * 80)
print("[6] XỬ LÝ CỘT COMPANY")
print("=" * 80)

company_missing = df["company"].isna().sum()
company_missing_percent = company_missing / len(df) * 100

print(
    f"company thiếu: {company_missing:,} dòng "
    f"({company_missing_percent:.2f}%)"
)

# company có tỷ lệ thiếu rất cao.
# Vì vậy không điền giá trị giả vào company.
# Ta giữ thông tin có/không có company bằng has_company,
# sau đó loại bỏ cột company.
df.drop(columns=["company"], inplace=True)

print("Đã loại bỏ cột company.")
print("Thông tin có company được giữ lại qua cột has_company.")


# ============================================================
# 10. XỬ LÝ CỘT AGENT
# ============================================================

print("\n" + "=" * 80)
print("[7] XỬ LÝ CỘT AGENT")
print("=" * 80)

agent_missing = df["agent"].isna().sum()

print(f"agent thiếu trước xử lý: {agent_missing:,}")

# Agent là mã đại lý.
# Giá trị 0 được dùng để biểu diễn trường hợp không có agent.
df["agent"] = df["agent"].fillna(0)

print(
    f"agent thiếu sau xử lý: {df['agent'].isna().sum():,}"
)


# ============================================================
# 11. XỬ LÝ CỘT CHILDREN
# ============================================================

print("\n" + "=" * 80)
print("[8] XỬ LÝ CỘT CHILDREN")
print("=" * 80)

children_missing = df["children"].isna().sum()

print(f"children thiếu trước xử lý: {children_missing:,}")

children_mode = df["children"].mode(dropna=True)[0]
children_median = df["children"].median()

print(f"Mode của children: {children_mode}")
print(f"Median của children: {children_median}")

# Nếu cả mode và median đều bằng 0,
# dùng 0 để thay thế giá trị thiếu.
if children_mode == 0 and children_median == 0:
    children_fill_value = 0
else:
    children_fill_value = children_median

df["children"] = df["children"].fillna(
    children_fill_value
)

print(
    f"Đã thay giá trị thiếu children bằng: "
    f"{children_fill_value}"
)

print(
    f"children thiếu sau xử lý: "
    f"{df['children'].isna().sum():,}"
)


# ============================================================
# 12. XỬ LÝ CỘT COUNTRY
# ============================================================

print("\n" + "=" * 80)
print("[9] XỬ LÝ CỘT COUNTRY")
print("=" * 80)

country_missing = df["country"].isna().sum()

print(f"country thiếu trước xử lý: {country_missing:,}")

# Không nên xóa các booking chỉ vì thiếu quốc gia.
# Dùng Unknown để giữ lại các booking.
df["country"] = df["country"].fillna("Unknown")

print(
    f"country thiếu sau xử lý: "
    f"{df['country'].isna().sum():,}"
)


# ============================================================
# 13. KIỂM TRA MISSING SAU KHI XỬ LÝ
# ============================================================

print("\n" + "=" * 80)
print("[10] KIỂM TRA MISSING SAU XỬ LÝ")
print("=" * 80)

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0].sort_values(
    ascending=False
)

if len(missing_after) > 0:
    print("Các cột vẫn còn missing:")
    print(missing_after)
else:
    print("Không còn giá trị thiếu trong dữ liệu.")


# Lưu thống kê missing sau xử lý.
missing_after_df = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_after_df.columns = [
    "column",
    "missing_count"
]

missing_after_df["missing_percent"] = (
    missing_after_df["missing_count"]
    / len(df)
    * 100
)

missing_after_df = missing_after_df[
    missing_after_df["missing_count"] > 0
]

missing_after_path = (
    PROCESSED_DIR / "missing_values_after.csv"
)

missing_after_df.to_csv(
    missing_after_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. LOẠI BỎ DUPLICATE
# ============================================================

print("\n" + "=" * 80)
print("[11] LOẠI BỎ DÒNG TRÙNG LẶP")
print("=" * 80)

rows_before_duplicate = len(df)

df = df.drop_duplicates().copy()

rows_after_duplicate = len(df)

duplicates_removed = (
    rows_before_duplicate
    - rows_after_duplicate
)

print(
    f"Số dòng đã loại bỏ do trùng: "
    f"{duplicates_removed:,}"
)

print(
    f"Kích thước sau khi loại duplicate: "
    f"{df.shape}"
)


# ============================================================
# 15. TẠO BIẾN TOTAL_GUESTS VÀ TOTAL_NIGHTS
# ============================================================

print("\n" + "=" * 80)
print("[12] TẠO BIẾN MỚI")
print("=" * 80)

# Tổng số khách trong một booking.
df["total_guests"] = (
    df["adults"]
    + df["children"]
    + df["babies"]
)

# Tổng số đêm lưu trú.
df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

print("Đã tạo:")
print("- total_guests")
print("- total_nights")


# ============================================================
# 16. KIỂM TRA BẤT THƯỜNG LOGIC
# ============================================================

print("\n" + "=" * 80)
print("[13] KIỂM TRA BẤT THƯỜNG LOGIC")
print("=" * 80)

zero_guests = (df["total_guests"] == 0).sum()
zero_nights = (df["total_nights"] == 0).sum()
negative_adr = (df["adr"] < 0).sum()

print(f"Booking có total_guests = 0: {zero_guests:,}")
print(f"Booking có total_nights = 0: {zero_nights:,}")
print(f"Booking có ADR < 0: {negative_adr:,}")

# Các bản ghi không có khách hoặc không có đêm lưu trú
# không phù hợp với mục tiêu phân tích booking khách sạn.
# ADR âm cũng là dữ liệu không hợp lệ.
invalid_mask = (
    (df["total_guests"] == 0)
    | (df["total_nights"] == 0)
    | (df["adr"] < 0)
)

invalid_count = invalid_mask.sum()

df = df.loc[~invalid_mask].copy()

print(
    f"\nĐã loại bỏ {invalid_count:,} dòng bất thường logic."
)

print(f"Kích thước sau xử lý logic: {df.shape}")

# ADR = 0 vẫn được giữ lại.
# Đây có thể là booking có giá trị ADR bằng 0 trong dữ liệu.
print(
    "Lưu ý: ADR = 0 được giữ lại, "
    "chỉ loại bỏ ADR âm."
)


# ============================================================
# 17. CHUYỂN ĐỔI THÁNG
# ============================================================

print("\n" + "=" * 80)
print("[14] CHUẨN HÓA THÁNG")
print("=" * 80)

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

df["arrival_month"] = pd.Categorical(
    df["arrival_date_month"],
    categories=month_order,
    ordered=True
)

print("Đã tạo arrival_month dạng categorical có thứ tự.")


# ============================================================
# 18. TẠO CỘT ARRIVAL_DATE
# ============================================================

print("\n" + "=" * 80)
print("[15] TẠO NGÀY ĐẾN")
print("=" * 80)

# Dataset có năm, tháng và ngày riêng.
# Ghép lại thành một cột ngày để thuận tiện phân tích.
df["arrival_date"] = pd.to_datetime(
    df["arrival_date_year"].astype(str)
    + "-"
    + df["arrival_date_month"].astype(str)
    + "-"
    + df["arrival_date_day_of_month"].astype(str),
    format="%Y-%B-%d",
    errors="coerce"
)

invalid_dates = df["arrival_date"].isna().sum()

print(f"Số ngày không chuyển đổi được: {invalid_dates:,}")


# ============================================================
# 19. THỐNG KÊ MÔ TẢ TRƯỚC XỬ LÝ OUTLIER
# ============================================================

print("\n" + "=" * 80)
print("[16] THỐNG KÊ MÔ TẢ TRƯỚC OUTLIER")
print("=" * 80)

numeric_columns = [
    "lead_time",
    "adr",
    "total_nights",
    "total_guests",
    "days_in_waiting_list",
]

stats_before_outlier = df[numeric_columns].describe().T

print(stats_before_outlier)

stats_before_path = (
    PROCESSED_DIR
    / "descriptive_statistics_before_outlier.csv"
)

stats_before_outlier.to_csv(
    stats_before_path,
    encoding="utf-8-sig"
)


# ============================================================
# 20. PHÁT HIỆN OUTLIER BẰNG IQR
# ============================================================

print("\n" + "=" * 80)
print("[17] PHÁT HIỆN OUTLIER BẰNG IQR")
print("=" * 80)


def iqr_info(series):
    """
    Tính Q1, Q3, IQR và giới hạn outlier theo phương pháp IQR.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_count = (
        (series < lower)
        | (series > upper)
    ).sum()

    return q1, q3, iqr, lower, upper, outlier_count


outlier_rows = []

for col in numeric_columns:

    q1, q3, iqr, lower, upper, count = iqr_info(
        df[col]
    )

    outlier_rows.append({
        "column": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count,
        "outlier_percent": count / len(df) * 100,
    })

outlier_df = pd.DataFrame(outlier_rows)

print(outlier_df)

outlier_path = (
    PROCESSED_DIR
    / "outlier_detection_iqr.csv"
)

outlier_df.to_csv(
    outlier_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. BOXplot TRƯỚC KHI XỬ LÝ OUTLIER
# ============================================================

print("\n" + "=" * 80)
print("[18] VẼ BOXPLOT TRƯỚC OUTLIER")
print("=" * 80)

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=df[
        [
            "adr",
            "lead_time",
            "total_nights",
            "total_guests",
        ]
    ]
)

plt.title(
    "Boxplot các biến số trước khi xử lý outlier",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Biến")
plt.ylabel("Giá trị")

plt.tight_layout()

before_boxplot_path = (
    FIGURE_DIR
    / "boxplot_before_outlier.png"
)

plt.savefig(
    before_boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Đã lưu: {before_boxplot_path}")


# ============================================================
# 22. XỬ LÝ ADR CỰC ĐOAN
# ============================================================

print("\n" + "=" * 80)
print("[19] XỬ LÝ ADR CỰC ĐOAN")
print("=" * 80)

# Đây là ngưỡng nghiệp vụ của bài.
# Không phải ngưỡng IQR.
#
# ADR >= 1000 được xem là giá trị cực đoan rất lớn
# và sẽ được loại khỏi dữ liệu dùng cho phân tích.
ADR_EXTREME_THRESHOLD = 1000

adr_extreme_mask = (
    df["adr"] >= ADR_EXTREME_THRESHOLD
)

adr_extreme_count = adr_extreme_mask.sum()

print(
    f"Số booking có ADR >= "
    f"{ADR_EXTREME_THRESHOLD}: "
    f"{adr_extreme_count:,}"
)

# Lưu các bản ghi ADR cực đoan trước khi xóa.
adr_extreme_records = df.loc[
    adr_extreme_mask
].copy()

adr_extreme_path = (
    PROCESSED_DIR
    / "adr_extreme_records.csv"
)

adr_extreme_records.to_csv(
    adr_extreme_path,
    index=False,
    encoding="utf-8-sig"
)

# Loại ADR cực đoan.
df = df.loc[~adr_extreme_mask].copy()

print(
    f"Đã loại bỏ {adr_extreme_count:,} "
    "booking có ADR cực đoan."
)

print(
    "Lưu ý: ngưỡng ADR >= 1000 là "
    "quy tắc xử lý riêng của bài, "
    "không phải giới hạn IQR."
)


# ============================================================
# 23. TẠO CÁC CỘT CLEAN
# ============================================================

print("\n" + "=" * 80)
print("[20] XỬ LÝ OUTLIER")
print("=" * 80)

# Dùng hệ số 3.0 để hạn chế ảnh hưởng của các giá trị quá lớn
# nhưng vẫn giữ lại những booking hợp lệ.
treatment_k = 3.0


def cap_by_iqr(series, k=3.0):
    """
    Giới hạn giá trị theo:
        Q1 - k*IQR
        Q3 + k*IQR

    Phương pháp này giữ lại dòng dữ liệu,
    chỉ giới hạn các giá trị quá lớn hoặc quá nhỏ.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return series.clip(
        lower=lower,
        upper=upper
    )


# ADR:
# Sau khi đã loại ADR >= 1000,
# tiếp tục giới hạn những giá trị ADR còn quá lớn theo IQR x 3.
df["adr_clean"] = cap_by_iqr(
    df["adr"],
    treatment_k
)

# Lead time:
# Không loại booking, chỉ giới hạn giá trị cực lớn.
df["lead_time_clean"] = cap_by_iqr(
    df["lead_time"],
    treatment_k
)

# Total nights:
# Không loại booking, chỉ giới hạn giá trị cực lớn.
df["total_nights_clean"] = cap_by_iqr(
    df["total_nights"],
    treatment_k
)

# Total guests:
# Không xử lý thêm vì số khách đã được kiểm tra logic.
df["total_guests_clean"] = df["total_guests"]

# Waiting list:
# Giữ nguyên vì có thể có giá trị cao nhưng vẫn mang ý nghĩa.
df["days_in_waiting_list_clean"] = (
    df["days_in_waiting_list"]
)


print("Đã tạo các cột:")
print("- adr_clean")
print("- lead_time_clean")
print("- total_nights_clean")
print("- total_guests_clean")
print("- days_in_waiting_list_clean")


# ============================================================
# 24. THỐNG KÊ SAU XỬ LÝ OUTLIER
# ============================================================

print("\n" + "=" * 80)
print("[21] THỐNG KÊ SAU XỬ LÝ OUTLIER")
print("=" * 80)

clean_numeric_columns = [
    "lead_time_clean",
    "adr_clean",
    "total_nights_clean",
    "total_guests_clean",
    "days_in_waiting_list_clean",
]

stats_after_outlier = (
    df[clean_numeric_columns]
    .describe()
    .T
)

print(stats_after_outlier)

stats_after_path = (
    PROCESSED_DIR
    / "descriptive_statistics_after_outlier.csv"
)

stats_after_outlier.to_csv(
    stats_after_path,
    encoding="utf-8-sig"
)


# ============================================================
# 25. BOXPLOT SAU KHI XỬ LÝ OUTLIER
# ============================================================

print("\n" + "=" * 80)
print("[22] VẼ BOXPLOT SAU OUTLIER")
print("=" * 80)

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=df[
        [
            "adr_clean",
            "lead_time_clean",
            "total_nights_clean",
            "total_guests_clean",
        ]
    ]
)

plt.title(
    "Boxplot các biến số sau khi xử lý outlier",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Biến")
plt.ylabel("Giá trị")

plt.tight_layout()

after_boxplot_path = (
    FIGURE_DIR
    / "boxplot_after_outlier.png"
)

plt.savefig(
    after_boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Đã lưu: {after_boxplot_path}")


# ============================================================
# 26. NHẬN XÉT VỀ QUÁ TRÌNH LÀM SẠCH
# ============================================================

print("\n" + "=" * 80)
print("[23] NHẬN XÉT SAU KHI LÀM SẠCH")
print("=" * 80)

print(
    f"Số dòng ban đầu: "
    f"{rows_before_duplicate:,}"
)

print(
    f"Số dòng hiện tại: "
    f"{len(df):,}"
)

print(
    f"Số dòng đã giảm: "
    f"{rows_before_duplicate - len(df):,}"
)

print("\nCác bước làm sạch chính:")
print("1. Xử lý giá trị thiếu.")
print("2. Xóa dòng trùng lặp.")
print("3. Tạo total_guests và total_nights.")
print("4. Loại các booking có số khách bằng 0.")
print("5. Loại các booking có số đêm bằng 0.")
print("6. Loại ADR âm.")
print("7. Tạo arrival_date.")
print("8. Phát hiện outlier bằng IQR.")
print("9. Loại ADR cực đoan >= 1000.")
print("10. Giới hạn một số outlier bằng IQR x 3.")


# ============================================================
# 27. CẤU HÌNH BIỂU ĐỒ EDA
# ============================================================

print("\n" + "=" * 80)
print("[24] BẮT ĐẦU EDA - VISUALIZATION")
print("=" * 80)

# Thiết lập giao diện chung cho biểu đồ.
sns.set_theme(style="whitegrid")

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# ============================================================
# 28. TỶ LỆ HỦY BOOKING
# ============================================================

print("\n" + "=" * 80)
print("[25] EDA 1 - CANCELLATION RATE")
print("=" * 80)

overall_cancellation_rate = (
    df["is_canceled"].mean() * 100
)

cancellation_summary = pd.DataFrame({
    "status": ["Not canceled", "Canceled"],
    "count": [
        (df["is_canceled"] == 0).sum(),
        (df["is_canceled"] == 1).sum(),
    ]
})

cancellation_summary["percentage"] = (
    cancellation_summary["count"]
    / len(df)
    * 100
)

print(
    f"Tỷ lệ booking bị hủy: "
    f"{overall_cancellation_rate:.2f}%"
)

print("\nPhân bố booking:")
print(cancellation_summary)

cancellation_summary.to_csv(
    PROCESSED_DIR / "cancellation_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# Vẽ biểu đồ.
plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=cancellation_summary,
    x="status",
    y="percentage"
)

plt.title(
    "Tỷ lệ Booking bị hủy",
    fontweight="bold"
)

plt.xlabel("Trạng thái")
plt.ylabel("Tỷ lệ (%)")

# Hiển thị số phần trăm trên đầu cột.
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_cancellation_rate.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. TỶ LỆ HỦY THEO KHÁCH SẠN
# ============================================================

print("\n" + "=" * 80)
print("[26] EDA 2 - CANCELLATION BY HOTEL")
print("=" * 80)

cancellation_by_hotel = (
    df.groupby("hotel")["is_canceled"]
    .agg(
        booking_count="count",
        cancellation_rate="mean"
    )
    .reset_index()
)

cancellation_by_hotel["cancellation_rate"] *= 100

print(cancellation_by_hotel)

cancellation_by_hotel.to_csv(
    PROCESSED_DIR / "cancellation_by_hotel.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=cancellation_by_hotel,
    x="hotel",
    y="cancellation_rate"
)

plt.title(
    "Tỷ lệ hủy Booking theo loại khách sạn",
    fontweight="bold"
)

plt.xlabel("Khách sạn")
plt.ylabel("Tỷ lệ hủy (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_cancellation_by_hotel.png",
    bbox_inches="tight"
)

plt.close()

hotel_highest_cancel = cancellation_by_hotel.loc[
    cancellation_by_hotel["cancellation_rate"].idxmax()
]

print(
    f"\nKhách sạn có tỷ lệ hủy cao hơn trong dữ liệu: "
    f"{hotel_highest_cancel['hotel']} "
    f"({hotel_highest_cancel['cancellation_rate']:.2f}%)"
)


# ============================================================
# 30. SỐ LƯỢNG BOOKING THEO THÁNG
# ============================================================

print("\n" + "=" * 80)
print("[27] EDA 3 - BOOKINGS BY MONTH")
print("=" * 80)

bookings_by_month = (
    df.groupby(
        "arrival_month",
        observed=False
    )
    .size()
    .reindex(month_order)
    .reset_index(name="booking_count")
)

print(bookings_by_month)

bookings_by_month.to_csv(
    PROCESSED_DIR / "bookings_by_month.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=bookings_by_month,
    x="arrival_month",
    y="booking_count"
)

plt.title(
    "Số lượng Booking theo tháng",
    fontweight="bold"
)

plt.xlabel("Tháng")
plt.ylabel("Số lượng Booking")

plt.xticks(rotation=30)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=2,
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_bookings_by_month.png",
    bbox_inches="tight"
)

plt.close()

peak_month = bookings_by_month.loc[
    bookings_by_month["booking_count"].idxmax()
]

print(
    f"\nTháng có nhiều booking nhất: "
    f"{peak_month['arrival_month']} "
    f"({int(peak_month['booking_count']):,} booking)"
)


# ============================================================
# 31. ADR THEO THÁNG VÀ KHÁCH SẠN
# ============================================================

print("\n" + "=" * 80)
print("[28] EDA 4 - ADR BY MONTH AND HOTEL")
print("=" * 80)

adr_by_month_hotel = (
    df.groupby(
        ["arrival_month", "hotel"],
        observed=False
    )["adr_clean"]
    .mean()
    .reset_index()
)

adr_by_month_hotel = (
    adr_by_month_hotel
    .dropna(subset=["adr_clean"])
)

print(adr_by_month_hotel)

adr_by_month_hotel.to_csv(
    PROCESSED_DIR / "adr_by_month_hotel.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=adr_by_month_hotel,
    x="arrival_month",
    y="adr_clean",
    hue="hotel",
    marker="o",
    linewidth=2
)

plt.title(
    "ADR trung bình theo tháng và loại khách sạn",
    fontweight="bold"
)

plt.xlabel("Tháng")
plt.ylabel("ADR trung bình")

plt.xticks(rotation=30)

plt.legend(
    title="Khách sạn"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_adr_by_month_hotel.png",
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 32. PHÂN PHỐI CÁC BIẾN SỐ
# ============================================================

print("\n" + "=" * 80)
print("[29] EDA 5 - NUMERIC DISTRIBUTIONS")
print("=" * 80)

distribution_columns = [
    "adr_clean",
    "lead_time_clean",
    "total_nights_clean",
    "total_guests_clean",
]

distribution_titles = {
    "adr_clean": "Phân phối ADR",
    "lead_time_clean": "Phân phối Lead Time",
    "total_nights_clean": "Phân phối Total Nights",
    "total_guests_clean": "Phân phối Total Guests",
}

# Chỉ tạo một figure gồm 4 ô.
# Như vậy không tạo quá nhiều file hình riêng lẻ.
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

axes = axes.flatten()

for ax, col in zip(
    axes,
    distribution_columns
):

    sns.histplot(
        data=df,
        x=col,
        kde=True,
        ax=ax
    )

    mean_value = df[col].mean()
    median_value = df[col].median()

    # Đường mean.
    ax.axvline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_value:.2f}"
    )

    # Đường median.
    ax.axvline(
        median_value,
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {median_value:.2f}"
    )

    ax.set_title(
        distribution_titles[col],
        fontweight="bold"
    )

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

    ax.legend(
        fontsize=8
    )

plt.suptitle(
    "Phân phối các biến số chính",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_numeric_distributions.png",
    bbox_inches="tight"
)

plt.close()

print(
    "Đã tạo 1 hình gồm 4 biểu đồ phân phối "
    "để tránh tạo quá nhiều hình."
)


# ============================================================
# 33. TOP 10 QUỐC GIA
# ============================================================

print("\n" + "=" * 80)
print("[30] EDA 6 - TOP 10 COUNTRIES")
print("=" * 80)

top10_countries = (
    df["country"]
    .value_counts()
    .head(10)
    .reset_index()
)

top10_countries.columns = [
    "country",
    "booking_count"
]

print(top10_countries)

top10_countries.to_csv(
    PROCESSED_DIR / "top10_countries.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 7))

top10_plot = top10_countries.sort_values(
    "booking_count"
)

ax = sns.barplot(
    data=top10_plot,
    x="booking_count",
    y="country"
)

plt.title(
    "Top 10 quốc gia có nhiều Booking nhất",
    fontweight="bold"
)

plt.xlabel("Số lượng Booking")
plt.ylabel("Quốc gia")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.0f",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_top10_countries.png",
    bbox_inches="tight"
)

plt.close()

print(
    f"\nQuốc gia có nhiều booking nhất: "
    f"{top10_countries.iloc[0]['country']}"
)


# ============================================================
# 34. TỶ LỆ HỦY THEO MARKET SEGMENT
# ============================================================

print("\n" + "=" * 80)
print("[31] EDA 7 - CANCELLATION BY MARKET SEGMENT")
print("=" * 80)

cancellation_by_segment = (
    df.groupby("market_segment")["is_canceled"]
    .agg(
        booking_count="count",
        cancellation_rate="mean"
    )
    .reset_index()
)

cancellation_by_segment["cancellation_rate"] *= 100

cancellation_by_segment = (
    cancellation_by_segment
    .sort_values(
        "cancellation_rate",
        ascending=False
    )
)

print(cancellation_by_segment)

cancellation_by_segment.to_csv(
    PROCESSED_DIR / "cancellation_by_market_segment.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 7))

ax = sns.barplot(
    data=cancellation_by_segment,
    x="cancellation_rate",
    y="market_segment"
)

plt.title(
    "Tỷ lệ hủy Booking theo Market Segment",
    fontweight="bold"
)

plt.xlabel("Tỷ lệ hủy (%)")
plt.ylabel("Market Segment")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3,
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_cancellation_market_segment.png",
    bbox_inches="tight"
)

plt.close()

highest_segment = cancellation_by_segment.iloc[0]

print(
    f"\nMarket segment có tỷ lệ hủy cao nhất "
    f"trong dữ liệu: "
    f"{highest_segment['market_segment']} "
    f"({highest_segment['cancellation_rate']:.2f}%)"
)


# ============================================================
# 35. TỶ LỆ HỦY THEO DEPOSIT TYPE
# ============================================================

print("\n" + "=" * 80)
print("[32] EDA 8 - CANCELLATION BY DEPOSIT TYPE")
print("=" * 80)

cancellation_by_deposit = (
    df.groupby("deposit_type")["is_canceled"]
    .agg(
        booking_count="count",
        cancellation_rate="mean"
    )
    .reset_index()
)

cancellation_by_deposit["cancellation_rate"] *= 100

print(cancellation_by_deposit)

cancellation_by_deposit.to_csv(
    PROCESSED_DIR / "cancellation_by_deposit_type.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 6))

ax = sns.barplot(
    data=cancellation_by_deposit,
    x="deposit_type",
    y="cancellation_rate"
)

plt.title(
    "Tỷ lệ hủy Booking theo Deposit Type",
    fontweight="bold"
)

plt.xlabel("Deposit Type")
plt.ylabel("Tỷ lệ hủy (%)")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_cancellation_deposit_type.png",
    bbox_inches="tight"
)

plt.close()


highest_deposit = cancellation_by_deposit.loc[
    cancellation_by_deposit["cancellation_rate"].idxmax()
]

print(
    f"\nDeposit type có tỷ lệ hủy cao nhất "
    f"trong dữ liệu: "
    f"{highest_deposit['deposit_type']} "
    f"({highest_deposit['cancellation_rate']:.2f}%)"
)


# ============================================================
# 36. KIỂM TRA CÁC FILE HÌNH ĐÃ TẠO
# ============================================================

print("\n" + "=" * 80)
print("[33] KIỂM TRA FILE HÌNH")
print("=" * 80)

figure_files = [
    "01_cancellation_rate.png",
    "02_cancellation_by_hotel.png",
    "03_bookings_by_month.png",
    "04_adr_by_month_hotel.png",
    "05_numeric_distributions.png",
    "06_top10_countries.png",
    "07_cancellation_market_segment.png",
    "08_cancellation_deposit_type.png",
    "boxplot_before_outlier.png",
    "boxplot_after_outlier.png",
]

existing_figures = []
missing_figures = []

for filename in figure_files:

    path = FIGURE_DIR / filename

    if path.exists():
        existing_figures.append(filename)
    else:
        missing_figures.append(filename)


print(
    f"Số file hình đã tạo: "
    f"{len(existing_figures)}/{len(figure_files)}"
)

if existing_figures:

    print("\nCác file đã có:")

    for filename in existing_figures:
        print(f"  ✓ {filename}")

if missing_figures:

    print("\nCác file chưa có:")

    for filename in missing_figures:
        print(f"  ✗ {filename}")

else:

    print(
        "\nTất cả file hình đều đã được tạo thành công."
    )


# ============================================================
# 37. KIỂM TRA MISSING CUỐI CÙNG
# ============================================================

print("\n" + "=" * 80)
print("[34] KIỂM TRA DỮ LIỆU CUỐI CÙNG")
print("=" * 80)

final_missing = df.isnull().sum()

final_missing_count = final_missing.sum()

print(
    f"Tổng số giá trị thiếu còn lại: "
    f"{final_missing_count:,}"
)

if final_missing_count == 0:
    print(
        "Kết quả: dữ liệu không còn giá trị thiếu."
    )
else:
    print(
        "Vẫn còn một số giá trị thiếu:"
    )
    print(
        final_missing[
            final_missing > 0
        ]
    )


# ============================================================
# 38. KIỂM TRA DUPLICATE CUỐI CÙNG
# ============================================================

print("\n" + "=" * 80)
print("[35] KIỂM TRA DUPLICATE CUỐI CÙNG")
print("=" * 80)

final_duplicates = df.duplicated().sum()

print(
    f"Số dòng trùng còn lại: "
    f"{final_duplicates:,}"
)

if final_duplicates == 0:
    print(
        "Kết quả: không còn dòng trùng hoàn toàn."
    )


# ============================================================
# 39. THỐNG KÊ CUỐI CÙNG
# ============================================================

print("\n" + "=" * 80)
print("[36] THỐNG KÊ DỮ LIỆU CUỐI CÙNG")
print("=" * 80)

print(
    f"Số dòng cuối cùng: "
    f"{len(df):,}"
)

print(
    f"Số cột cuối cùng: "
    f"{len(df.columns)}"
)

print(
    f"Kích thước cuối cùng: "
    f"{df.shape}"
)

print("\nMột số thống kê chính:")

print(
    f"- Tỷ lệ hủy booking: "
    f"{overall_cancellation_rate:.2f}%"
)

print(
    f"- ADR trung bình sau xử lý: "
    f"{df['adr_clean'].mean():.2f}"
)

print(
    f"- Lead time trung bình sau xử lý: "
    f"{df['lead_time_clean'].mean():.2f}"
)

print(
    f"- Số đêm trung bình: "
    f"{df['total_nights_clean'].mean():.2f}"
)

print(
    f"- Số khách trung bình: "
    f"{df['total_guests_clean'].mean():.2f}"
)


# ============================================================
# 40. LƯU DATASET CLEANED
# ============================================================

print("\n" + "=" * 80)
print("[37] LƯU DATASET SAU KHI LÀM SẠCH")
print("=" * 80)

cleaned_path = (
    PROCESSED_DIR
    / "hotel_bookings_cleaned.csv"
)

df.to_csv(
    cleaned_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"Đã lưu dataset cleaned tại:"
)

print(
    cleaned_path
)

print(
    f"Kích thước dataset đã lưu: "
    f"{df.shape}"
)


# ============================================================
# 41. TỔNG KẾT
# ============================================================

print("\n" + "=" * 80)
print("KẾT LUẬN TV1")
print("=" * 80)

print(
    "\n1. Dữ liệu Hotel Booking Demand đã được kiểm tra "
    "và làm sạch."
)

print(
    "2. Các giá trị thiếu được xử lý theo từng loại biến."
)

print(
    "3. Các dòng trùng lặp đã được loại bỏ."
)

print(
    "4. Các booking bất thường về số khách, số đêm "
    "và ADR âm đã được xử lý."
)

print(
    "5. Outlier được phát hiện bằng phương pháp IQR."
)

print(
    "6. ADR cực đoan >= 1000 được loại bỏ "
    "theo quy tắc xử lý của bài."
)

print(
    "7. Một số biến số được giới hạn outlier bằng IQR x 3 "
    "nhằm giảm ảnh hưởng của giá trị quá lớn."
)

print(
    "8. Các biểu đồ EDA chính đã được tạo để phân tích "
    "tỷ lệ hủy, booking theo tháng, ADR, quốc gia, "
    "market segment và deposit type."
)

print("\nCác file quan trọng:")

print(
    f"- Dataset cleaned:\n  {cleaned_path}"
)

print(
    f"- Thư mục biểu đồ:\n  {FIGURE_DIR}"
)

print(
    f"- Thư mục dữ liệu xử lý:\n  {PROCESSED_DIR}"
)

print("\n" + "=" * 80)
print("HOÀN THÀNH TV1 - EDA & DATA CLEANING")
print("=" * 80)