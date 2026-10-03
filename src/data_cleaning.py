"""
TV1 - DATA CLEANING
Dataset: Hotel Booking Demand

File này thực hiện:
1. Đọc dữ liệu gốc
2. Kiểm tra cấu trúc dữ liệu
3. Xử lý giá trị thiếu
4. Xóa dữ liệu trùng lặp
5. Kiểm tra dữ liệu không hợp lệ
6. Tạo các biến mới
7. Kiểm tra ngày tháng
8. Phát hiện outlier bằng IQR
9. Xử lý giá trị ADR cực đoan
10. Capping một số biến bằng IQR
11. Lưu dữ liệu sạch

Kết quả:
- Dữ liệu sạch: data/processed/hotel_bookings_cleaned.csv
- Các bảng thống kê: results/
"""

from pathlib import Path
import pandas as pd



# ============================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================================================

# BASE_DIR là thư mục gốc của project Hotel-Booking-Demand
BASE_DIR = Path(__file__).resolve().parent.parent

# File dữ liệu gốc
RAW_PATH = BASE_DIR / "data" / "raw" / "hotel_bookings.csv"

# Chỉ lưu file dữ liệu sạch vào thư mục processed
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Các bảng thống kê trong quá trình cleaning được lưu vào results
RESULTS_DIR = BASE_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================

print("=" * 70)
print("TV1 - DATA CLEANING")
print("=" * 70)

print("\n[1] Đọc dữ liệu...")

df = pd.read_csv(RAW_PATH)

# Lưu số dòng ban đầu để cuối chương trình có thể báo cáo
# dữ liệu đã giảm bao nhiêu dòng sau quá trình cleaning.
initial_rows = len(df)

print(f"Kích thước dữ liệu ban đầu: {df.shape}")
print(f"Số dòng ban đầu: {initial_rows:,}")
print(f"Số cột ban đầu: {df.shape[1]:,}")


# ============================================================
# 3. KIỂM TRA THÔNG TIN DỮ LIỆU
# ============================================================

print("\n[2] Thông tin dữ liệu:")
print(df.info())

print("\n5 dòng đầu tiên:")
print(df.head())


# ============================================================
# 4. KIỂM TRA GIÁ TRỊ THIẾU
# ============================================================

print("\n[3] KIỂM TRA GIÁ TRỊ THIẾU")

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]

print("\nGiá trị thiếu trước khi xử lý:")
print(missing_before)


# Lưu bảng missing trước xử lý vào results
missing_before_table = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_before_table.columns = ["column", "missing_before"]


# ============================================================
# 5. TẠO CỘT ĐÁNH DẤU MISSING
# ============================================================

# Các cột này được tạo để ghi nhận ban đầu dữ liệu có bị thiếu hay không.
# Sau khi điền dữ liệu, thông tin này vẫn được giữ lại để phục vụ phân tích.

df["agent_was_missing"] = df["agent"].isna().astype(int)

df["children_was_missing"] = df["children"].isna().astype(int)

df["country_was_missing"] = df["country"].isna().astype(int)

# company có rất nhiều giá trị thiếu.
# Tạo biến has_company trước khi loại bỏ cột company.
df["has_company"] = df["company"].notna().astype(int)


# ============================================================
# 6. XỬ LÝ CỘT COMPANY
# ============================================================

print("\n[4] Xử lý cột company...")

# company có tỷ lệ thiếu rất cao.
# Vì vậy giữ lại cột này và điền giá trị sẽ không mang nhiều thông tin.
# Thay vào đó, ta giữ lại thông tin có/không có company qua has_company
# rồi loại bỏ cột company.
company_missing_rate = df["company"].isna().mean() * 100

print(f"Tỷ lệ thiếu của company: {company_missing_rate:.2f}%")

df.drop(columns=["company"], inplace=True)

print("Đã loại bỏ cột company.")


# ============================================================
# 7. XỬ LÝ AGENT
# ============================================================

print("\n[5] Xử lý agent...")

# agent là mã đại lý.
# Giá trị thiếu được thay bằng 0 để biểu diễn trường hợp
# không có thông tin đại lý.
df["agent"] = df["agent"].fillna(0)

print("Đã thay giá trị thiếu của agent bằng 0.")


# ============================================================
# 8. XỬ LÝ CHILDREN
# ============================================================

print("\n[6] Xử lý children...")

children_mode = df["children"].mode(dropna=True)[0]
children_median = df["children"].median()

print(f"Mode của children: {children_mode}")
print(f"Median của children: {children_median}")

# children là số lượng trẻ em nên giá trị phải là số nguyên.
# Nếu mode và median đều bằng 0 thì 0 là giá trị đại diện hợp lý.
# Nếu không, dùng median để giảm ảnh hưởng của các giá trị lệch.
# Median có thể cho ra số thập phân, ví dụ 1.5,
# nên làm tròn về số nguyên gần nhất trước khi điền.
if children_mode == 0 and children_median == 0:
    children_fill_value = 0
else:
    children_fill_value = round(children_median)

df["children"] = df["children"].fillna(children_fill_value)

# Ép kiểu int để đảm bảo cột children chỉ chứa số nguyên.
df["children"] = df["children"].astype(int)

print(f"Đã thay giá trị thiếu của children bằng: {children_fill_value}")
# ============================================================
# 9. XỬ LÝ COUNTRY
# ============================================================

print("\n[7] Xử lý country...")

# country là quốc gia của khách.
# Nếu thiếu thì dùng "Unknown" để giữ lại dòng dữ liệu.
df["country"] = df["country"].fillna("Unknown")

print("Đã thay giá trị thiếu của country bằng 'Unknown'.")


# ============================================================
# 10. KIỂM TRA MISSING SAU XỬ LÝ
# ============================================================

missing_after = df.isnull().sum()

missing_after_nonzero = missing_after[missing_after > 0]

print("\nGiá trị thiếu sau khi xử lý:")

if len(missing_after_nonzero) == 0:
    print("Không còn giá trị thiếu.")
else:
    print(missing_after_nonzero)


# Tạo bảng so sánh missing trước và sau
missing_after_table = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_after_table.columns = ["column", "missing_after"]

missing_summary = pd.merge(
    missing_before_table,
    missing_after_table,
    on="column",
    how="outer"
).fillna(0)

missing_summary["missing_before"] = (
    missing_summary["missing_before"].astype(int)
)

missing_summary["missing_after"] = (
    missing_summary["missing_after"].astype(int)
)

missing_summary.to_csv(
    RESULTS_DIR / "missing_values_before_after.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nĐã lưu bảng missing: "
    "results/missing_values_before_after.csv"
)


# ============================================================
# 11. XÓA DỮ LIỆU TRÙNG LẶP
# ============================================================

print("\n[8] Kiểm tra dữ liệu trùng lặp...")

duplicate_count = df.duplicated().sum()

print(f"Số dòng trùng lặp: {duplicate_count:,}")

if duplicate_count > 0:
    df = df.drop_duplicates().copy()
    print(f"Đã xóa {duplicate_count:,} dòng trùng lặp.")
else:
    print("Không có dòng trùng lặp.")


# ============================================================
# 12. TẠO CÁC BIẾN MỚI
# ============================================================

print("\n[9] Tạo các biến mới...")

# Tổng số khách = người lớn + trẻ em + trẻ sơ sinh
df["total_guests"] = (
    df["adults"]
    + df["children"]
    + df["babies"]
)

# Tổng số đêm lưu trú
df["total_nights"] = (
    df["stays_in_weekend_nights"]
    + df["stays_in_week_nights"]
)

print("Đã tạo:")
print("- total_guests")
print("- total_nights")


# ============================================================
# 13. KIỂM TRA DỮ LIỆU KHÔNG HỢP LỆ
# ============================================================

print("\n[10] Kiểm tra dữ liệu không hợp lệ...")

invalid_guest_mask = df["total_guests"] == 0
invalid_night_mask = df["total_nights"] == 0
invalid_adr_mask = df["adr"] < 0

invalid_guest_count = invalid_guest_mask.sum()
invalid_night_count = invalid_night_mask.sum()
invalid_adr_count = invalid_adr_mask.sum()

print(f"Số dòng total_guests = 0: {invalid_guest_count:,}")
print(f"Số dòng total_nights = 0: {invalid_night_count:,}")
print(f"Số dòng ADR < 0: {invalid_adr_count:,}")

# total_guests = 0 nghĩa là không có người nào trong booking.
# total_nights = 0 nghĩa là không có đêm lưu trú.
# ADR < 0 là giá trị không hợp lệ.
invalid_mask = (
    invalid_guest_mask
    | invalid_night_mask
    | invalid_adr_mask
)

invalid_total = invalid_mask.sum()

if invalid_total > 0:
    df = df.loc[~invalid_mask].copy()
    print(f"Đã loại {invalid_total:,} dòng dữ liệu không hợp lệ.")
else:
    print("Không có dòng dữ liệu không hợp lệ cần loại bỏ.")


# ============================================================
# 14. XỬ LÝ NGÀY ĐẾN
# ============================================================

print("\n[11] Kiểm tra và chuyển đổi ngày đến...")

# Thứ tự tháng trong dataset là tên tháng bằng tiếng Anh.
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
    "December"
]

# Chuyển tên tháng sang số tháng.
month_number = {
    month: number
    for number, month in enumerate(month_order, start=1)
}

df["arrival_month_number"] = (
    df["arrival_date_month"].map(month_number)
)

# Tạo ngày hoàn chỉnh từ năm, tháng và ngày.
# errors="coerce" sẽ chuyển ngày không hợp lệ thành NaT.
df["arrival_date"] = pd.to_datetime(
    {
        "year": df["arrival_date_year"],
        "month": df["arrival_month_number"],
        "day": df["arrival_date_day_of_month"]
    },
    errors="coerce"
)

invalid_date_mask = df["arrival_date"].isna()
invalid_date_count = invalid_date_mask.sum()

print(f"Số ngày không chuyển đổi được: {invalid_date_count:,}")

if invalid_date_count > 0:

    # Lưu các dòng có ngày không hợp lệ để kiểm tra nguyên nhân.
    invalid_date_records = df.loc[
        invalid_date_mask,
        [
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_day_of_month"
        ]
    ].copy()

    invalid_date_records.to_csv(
        RESULTS_DIR / "invalid_arrival_dates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Đã lưu các dòng ngày không hợp lệ: "
        "results/invalid_arrival_dates.csv"
    )

    # Sau khi kiểm tra, các dòng không có ngày hợp lệ
    # được loại bỏ vì không thể sử dụng chính xác cho phân tích theo thời gian.
    df = df.loc[~invalid_date_mask].copy()

    print(
        f"Đã loại {invalid_date_count:,} dòng do ngày không hợp lệ."
    )

else:
    print("Không có ngày không hợp lệ.")

# Cột này chỉ dùng trong quá trình chuyển đổi.
df.drop(columns=["arrival_month_number"], inplace=True)


# ============================================================
# 15. SẮP XẾP THỨ TỰ THÁNG
# ============================================================

# Chuyển arrival_date_month thành categorical để khi phân tích
# các tháng sẽ được sắp xếp từ January -> December thay vì alphabet.
df["arrival_date_month"] = pd.Categorical(
    df["arrival_date_month"],
    categories=month_order,
    ordered=True
)


# ============================================================
# 16. XỬ LÝ ADR CỰC ĐOAN
# ============================================================

print("\n[12] Kiểm tra ADR cực đoan...")

# ADR = Average Daily Rate.
# Đây là giá trung bình mỗi ngày của booking.
# Một số giá trị ADR quá lớn có thể làm ảnh hưởng mạnh đến phân tích.
ADR_EXTREME_THRESHOLD = 1000

adr_extreme_mask = df["adr"] >= ADR_EXTREME_THRESHOLD

adr_extreme_count = adr_extreme_mask.sum()

print(
    f"Số dòng ADR >= {ADR_EXTREME_THRESHOLD}: "
    f"{adr_extreme_count:,}"
)

if adr_extreme_count > 0:

    # Lưu lại các dòng cực đoan để có thể kiểm tra.
    df.loc[adr_extreme_mask].to_csv(
        RESULTS_DIR / "adr_extreme_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Đã lưu các dòng ADR cực đoan: "
        "results/adr_extreme_records.csv"
    )

    # Đây là quy tắc xử lý dữ liệu của bài:
    # loại các booking có ADR >= 1000 trước khi capping.
    df = df.loc[~adr_extreme_mask].copy()

    print(
        f"Đã loại {adr_extreme_count:,} dòng ADR cực đoan."
    )

else:
    print("Không có ADR cực đoan.")


# ============================================================
# 17. PHÁT HIỆN OUTLIER BẰNG IQR
# ============================================================

print("\n[13] PHÁT HIỆN OUTLIER BẰNG IQR")

# IQR = Interquartile Range.
#
# Q1 là phân vị thứ 25:
# 25% dữ liệu nhỏ hơn hoặc bằng Q1.
#
# Q3 là phân vị thứ 75:
# 75% dữ liệu nhỏ hơn hoặc bằng Q3.
#
# IQR = Q3 - Q1.
# IQR đại diện cho khoảng chứa 50% dữ liệu ở giữa.
#
# Theo quy tắc IQR phổ biến:
# Lower Bound = Q1 - 1.5 * IQR
# Upper Bound = Q3 + 1.5 * IQR
#
# Giá trị nằm ngoài hai ngưỡng trên được xem là outlier
# trong bước phát hiện.
#
# Lưu ý:
# Phát hiện outlier KHÔNG có nghĩa là bắt buộc phải xóa.
# Sau khi phát hiện, ta sẽ quyết định cách xử lý phù hợp.

IQR_DETECTION_K = 1.5


def iqr_info(series):
    """
    Tính các thông tin cần thiết để phát hiện outlier bằng IQR.
    """

    # Q1: phân vị 25%
    q1 = series.quantile(0.25)

    # Q3: phân vị 75%
    q3 = series.quantile(0.75)

    # IQR là khoảng giữa Q1 và Q3.
    iqr = q3 - q1

    # Hai ngưỡng dùng để phát hiện outlier.
    lower = q1 - IQR_DETECTION_K * iqr
    upper = q3 + IQR_DETECTION_K * iqr

    # Đếm số giá trị nằm ngoài hai ngưỡng.
    outlier_count = (
        (series < lower)
        | (series > upper)
    ).sum()

    return q1, q3, iqr, lower, upper, outlier_count


outlier_columns = [
    "lead_time",
    "adr",
    "total_nights",
    "total_guests",
    "days_in_waiting_list"
]

outlier_results = []

for column in outlier_columns:

    q1, q3, iqr, lower, upper, outlier_count = iqr_info(
        df[column]
    )

    outlier_results.append(
        {
            "column": column,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count
        }
    )

    print(f"\n{column}:")
    print(f"  Q1 = {q1:.2f}")
    print(f"  Q3 = {q3:.2f}")
    print(f"  IQR = {iqr:.2f}")
    print(f"  Lower bound = {lower:.2f}")
    print(f"  Upper bound = {upper:.2f}")
    print(f"  Số outlier = {outlier_count:,}")


outlier_summary = pd.DataFrame(outlier_results)

outlier_summary.to_csv(
    RESULTS_DIR / "outlier_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nĐã lưu bảng outlier: "
    "results/outlier_summary.csv"
)


# ============================================================
# 18. CAPPING OUTLIER
# ============================================================

print("\n[14] XỬ LÝ OUTLIER BẰNG CAPPING")

# Ở bước phát hiện, sử dụng 1.5 * IQR.
#
# Khi xử lý, dùng k = 3.0 để capping bảo thủ hơn.
# Nghĩa là chỉ giới hạn các giá trị cực đoan hơn:
#
# Lower = Q1 - 3 * IQR
# Upper = Q3 + 3 * IQR
#
# Các giá trị được giữ lại nhưng nếu vượt quá ngưỡng
# thì sẽ được đưa về đúng ngưỡng.
#
# Capping KHÔNG xóa dòng dữ liệu.

TREATMENT_K = 3.0


def cap_by_iqr(series, k=3.0):
    """
    Capping dữ liệu bằng ngưỡng IQR.

    Trả về:
    - Series sau khi capping
    - Số giá trị thực sự bị thay đổi
    """

    # Tính Q1 và Q3
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    # Tính IQR
    iqr = q3 - q1

    # Tính ngưỡng capping.
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    # Xác định các giá trị vượt khỏi ngưỡng.
    changed_mask = (
        (series < lower)
        | (series > upper)
    )

    # Số giá trị thực sự bị thay đổi.
    changed_count = changed_mask.sum()

    # clip giữ nguyên giá trị bình thường,
    # chỉ giới hạn các giá trị nằm ngoài khoảng.
    cleaned_series = series.clip(
        lower=lower,
        upper=upper
    )

    return cleaned_series, changed_count, lower, upper


# ADR
df["adr_clean"], adr_capped_count, adr_lower, adr_upper = (
    cap_by_iqr(df["adr"], TREATMENT_K)
)

# Lead time
(
    df["lead_time_clean"],
    lead_time_capped_count,
    lead_time_lower,
    lead_time_upper
) = cap_by_iqr(
    df["lead_time"],
    TREATMENT_K
)

# Total nights
(
    df["total_nights_clean"],
    total_nights_capped_count,
    total_nights_lower,
    total_nights_upper
) = cap_by_iqr(
    df["total_nights"],
    TREATMENT_K
)

# Với total_guests và days_in_waiting_list,
# giữ nguyên dữ liệu vì không áp dụng capping trong bài.
df["total_guests_clean"] = df["total_guests"]

df["days_in_waiting_list_clean"] = (
    df["days_in_waiting_list"]
)


print("\nKẾT QUẢ CAPPING:")

print(
    f"- ADR: {adr_capped_count:,} giá trị bị capping"
)

print(
    f"- lead_time: {lead_time_capped_count:,} "
    "giá trị bị capping"
)

print(
    f"- total_nights: {total_nights_capped_count:,} "
    "giá trị bị capping"
)

print(
    "- total_guests: không capping"
)

print(
    "- days_in_waiting_list: không capping"
)


# Tạo bảng thống kê capping.
capping_summary = pd.DataFrame(
    [
        {
            "column": "adr",
            "method": "IQR capping",
            "k": TREATMENT_K,
            "lower_bound": adr_lower,
            "upper_bound": adr_upper,
            "capped_count": adr_capped_count
        },
        {
            "column": "lead_time",
            "method": "IQR capping",
            "k": TREATMENT_K,
            "lower_bound": lead_time_lower,
            "upper_bound": lead_time_upper,
            "capped_count": lead_time_capped_count
        },
        {
            "column": "total_nights",
            "method": "IQR capping",
            "k": TREATMENT_K,
            "lower_bound": total_nights_lower,
            "upper_bound": total_nights_upper,
            "capped_count": total_nights_capped_count
        }
    ]
)

capping_summary["capped_percent"] = (
    capping_summary["capped_count"]
    / len(df)
    * 100
)

capping_summary.to_csv(
    RESULTS_DIR / "capping_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nĐã lưu bảng capping: "
    "results/capping_summary.csv"
)


# ============================================================
# 19. CHỌN CÁC CỘT CUỐI CÙNG
# ============================================================

# Giữ lại các cột clean để EDA sử dụng.
# Các cột *_clean giúp phân biệt dữ liệu gốc và dữ liệu sau xử lý.
#
# Có thể đổi tên các cột clean về tên ban đầu để dataset cuối
# dễ sử dụng hơn.

df["adr"] = df["adr_clean"]

df["lead_time"] = df["lead_time_clean"]

df["total_nights"] = df["total_nights_clean"]

df["total_guests"] = df["total_guests_clean"]

df["days_in_waiting_list"] = (
    df["days_in_waiting_list_clean"]
)

# Sau khi thay thế dữ liệu gốc bằng dữ liệu đã xử lý,
# các cột tạm thời *_clean không cần thiết nữa.
df.drop(
    columns=[
        "adr_clean",
        "lead_time_clean",
        "total_nights_clean",
        "total_guests_clean",
        "days_in_waiting_list_clean"
    ],
    inplace=True
)


# ============================================================
# 20. KIỂM TRA CUỐI
# ============================================================

print("\n[15] KIỂM TRA DỮ LIỆU SAU CLEANING")

print(f"Số dòng ban đầu: {initial_rows:,}")
print(f"Số dòng sau cleaning: {len(df):,}")

print(
    f"Số dòng đã giảm: "
    f"{initial_rows - len(df):,}"
)

print(f"Số cột sau cleaning: {df.shape[1]:,}")

print(
    f"Số giá trị thiếu còn lại: "
    f"{df.isnull().sum().sum():,}"
)

print(
    f"Số dòng trùng lặp còn lại: "
    f"{df.duplicated().sum():,}"
)


# ============================================================
# 21. KIỂM TRA CÁC ĐIỀU KIỆN DỮ LIỆU HỢP LỆ
# ============================================================

# Kiểm tra giá trị children có phải số nguyên không.
if not (df["children"] % 1 == 0).all():
    raise ValueError(
        "Cột children vẫn còn giá trị không nguyên!"
    )

# Kiểm tra dữ liệu không hợp lệ.
if (df["adr"] < 0).any():
    raise ValueError("Vẫn còn ADR âm!")

if (df["total_guests"] <= 0).any():
    raise ValueError("Vẫn còn booking không có khách!")

if (df["total_nights"] <= 0).any():
    raise ValueError("Vẫn còn booking có 0 đêm!")

# Kiểm tra ngày tháng.
if df["arrival_date"].isna().any():
    raise ValueError("Vẫn còn ngày đến không hợp lệ!")

print("\nTất cả kiểm tra cuối đã hoàn thành.")


# ============================================================
# 22. LƯU DỮ LIỆU SẠCH
# ============================================================

CLEANED_PATH = (
    PROCESSED_DIR / "hotel_bookings_cleaned.csv"
)


# ============================================================
# 23. KẾT LUẬN DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("KẾT LUẬN DATA CLEANING")
print("=" * 70)

print(
    f"- Dữ liệu ban đầu có {initial_rows:,} dòng."
)

print(
    f"- Dữ liệu sau cleaning còn {len(df):,} dòng."
)

print(
    f"- Đã xử lý các giá trị thiếu của agent, children và country."
)

print(
    f"- Đã loại bỏ cột company do tỷ lệ thiếu cao."
)

print(
    f"- Đã xóa {duplicate_count:,} dòng trùng lặp."
)

print(
    f"- Đã loại {invalid_total:,} dòng dữ liệu không hợp lệ."
)

print(
    f"- Đã kiểm tra {invalid_date_count:,} ngày không hợp lệ."
)

print(
    f"- Đã loại {adr_extreme_count:,} dòng ADR cực đoan."
)

print(
    f"- ADR bị capping: {adr_capped_count:,} giá trị."
)

print(
    f"- lead_time bị capping: {lead_time_capped_count:,} giá trị."
)

print(
    f"- total_nights bị capping: {total_nights_capped_count:,} giá trị."
)

print(
    "\nFile dữ liệu sạch:"
)

print(CLEANED_PATH)

print("\nCác bảng thống kê được lưu trong:")
print(RESULTS_DIR)

print("=" * 70)