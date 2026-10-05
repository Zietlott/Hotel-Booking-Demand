from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# EDA sử dụng dữ liệu đã được cleaning.
CLEANED_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "hotel_bookings_cleaned.csv"
)

RESULTS_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. ĐỌC DỮ LIỆU
# ============================================================
print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print("\n[1] Đọc dữ liệu sạch...")
df = pd.read_csv(CLEANED_PATH)

print(f"Kích thước dữ liệu: {df.shape}")
print(f"Số dòng: {len(df):,}")
print(f"Số cột: {len(df.columns):,}")

# Chuyển arrival_date về kiểu datetime để phân tích theo thời gian.
df["arrival_date"] = pd.to_datetime(
    df["arrival_date"],
    errors="coerce"
)

# ============================================================
# 3. THỐNG KÊ MÔ TẢ
# ============================================================
print("\n[2] THỐNG KÊ MÔ TẢ")
print("\nCác biến số:")
print(df.describe())
print("\nCác biến dạng category:")
print(df.describe(include="object"))

# ============================================================
# 4. TỶ LỆ HỦY ĐẶT PHÒNG
# ============================================================
print("\n[3] PHÂN TÍCH TỶ LỆ HỦY")
cancellation_rate = (
    df["is_canceled"]
    .value_counts(normalize=True)
    * 100
)
not_cancelled = cancellation_rate.get(0, 0)
cancelled = cancellation_rate.get(1, 0)
print(f"Tỷ lệ không hủy: {not_cancelled:.2f}%")
print(f"Tỷ lệ hủy: {cancelled:.2f}%")

# ============================================================
# BIỂU ĐỒ 1
# ============================================================
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="is_canceled")
plt.title("Tỷ lệ Booking bị hủy và không bị hủy", fontsize=14)
plt.xlabel("Trạng thái booking")
plt.ylabel("Số lượng booking")
plt.xticks([0, 1], ["Không hủy", "Hủy"])
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "01_cancellation_overall.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print("\nNHẬN XÉT BIỂU ĐỒ 1:")
print(f"- Booking không bị hủy chiếm {not_cancelled:.2f}%.")
print(f"- Booking bị hủy chiếm {cancelled:.2f}%.")
print("- Biểu đồ cho thấy sự khác biệt về số lượng giữa hai trạng thái hủy và không hủy.")

# ============================================================
# 5. TỶ LỆ HỦY THEO HOTEL
# ============================================================
print("\n[4] TỶ LỆ HỦY THEO LOẠI KHÁCH SẠN")
hotel_cancellation = (
    df.groupby("hotel")["is_canceled"]
    .mean()
    .mul(100)
    .reset_index(name="cancellation_rate")
)
print(hotel_cancellation)

# ============================================================
# BIỂU ĐỒ 2
# ============================================================
plt.figure(figsize=(8, 6))
sns.barplot(
    data=hotel_cancellation,
    x="hotel",
    y="cancellation_rate"
)
plt.title("Tỷ lệ hủy booking theo loại khách sạn", fontsize=14)
plt.xlabel("Loại khách sạn")
plt.ylabel("Tỷ lệ hủy (%)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_cancellation_by_hotel.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

hotel_max = hotel_cancellation.loc[hotel_cancellation["cancellation_rate"].idxmax()]
hotel_min = hotel_cancellation.loc[hotel_cancellation["cancellation_rate"].idxmin()]

print("\nNHẬN XÉT BIỂU ĐỒ 2:")
print(
    f"- {hotel_max['hotel']} có tỷ lệ hủy "
    f"{hotel_max['cancellation_rate']:.2f}%."
)

print(
    f"- {hotel_min['hotel']} có tỷ lệ hủy "
    f"{hotel_min['cancellation_rate']:.2f}%."
)

print(
    f"- Chênh lệch tỷ lệ hủy giữa hai loại khách sạn là "
    f"{hotel_max['cancellation_rate'] - hotel_min['cancellation_rate']:.2f} "
    "điểm phần trăm."
)

# ============================================================
# 6. SỐ BOOKING THEO THÁNG
# ============================================================
print("\n[5] SỐ BOOKING THEO THÁNG")
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

monthly_bookings = (
    df["arrival_date_month"]
    .value_counts()
    .reindex(month_order)
)

print(monthly_bookings)

# ============================================================
# BIỂU ĐỒ 3
# ============================================================
plt.figure(figsize=(12, 6))
sns.barplot(
    x=monthly_bookings.index,
    y=monthly_bookings.values
)

plt.title("Số lượng Booking theo tháng", fontsize=14)
plt.xlabel("Tháng")
plt.ylabel("Số lượng booking")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "03_bookings_by_month.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

peak_month = monthly_bookings.idxmax()
peak_value = monthly_bookings.max()

lowest_month = monthly_bookings.idxmin()
lowest_value = monthly_bookings.min()

print("\nNHẬN XÉT BIỂU ĐỒ 3:")
print(f"- Tháng có nhiều booking nhất: {peak_month} ({peak_value:,} booking).")
print(f"- Tháng có ít booking nhất: {lowest_month} ({lowest_value:,} booking).")

# ============================================================
# 7. ADR THEO THÁNG VÀ HOTEL
# ============================================================

print("\n[6] ADR TRUNG BÌNH THEO THÁNG VÀ HOTEL")
adr_by_month_hotel = (
    df.groupby(
        ["arrival_date_month", "hotel"],
        observed=True
    )["adr"]
    .mean()
    .reset_index()
)
print(adr_by_month_hotel)

# ============================================================
# BIỂU ĐỒ 4
# ============================================================
plt.figure(figsize=(13, 6))
sns.lineplot(
    data=adr_by_month_hotel,
    x="arrival_date_month",
    y="adr",
    hue="hotel",
    marker="o"
)
plt.title(
    "ADR trung bình theo tháng và loại khách sạn",
    fontsize=14
)

plt.xlabel("Tháng")
plt.ylabel("ADR trung bình")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "04_adr_by_month_hotel.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

adr_max = adr_by_month_hotel.loc[adr_by_month_hotel["adr"].idxmax()]
adr_min = adr_by_month_hotel.loc[adr_by_month_hotel["adr"].idxmin()]

print("\nNHẬN XÉT BIỂU ĐỒ 4:")
print(
    f"- ADR trung bình cao nhất thuộc về "
    f"{adr_max['hotel']} vào {adr_max['arrival_date_month']}: "
    f"{adr_max['adr']:.2f}."
)

print(
    f"- ADR trung bình thấp nhất thuộc về "
    f"{adr_min['hotel']} vào {adr_min['arrival_date_month']}: "
    f"{adr_min['adr']:.2f}."
)

# ============================================================
# 8. PHÂN PHỐI CÁC BIẾN SỐ
# ============================================================
print("\n[7] PHÂN PHỐI CÁC BIẾN SỐ")
numeric_columns = [
    "lead_time",
    "adr",
    "total_nights",
    "total_guests"
]

for column in numeric_columns:
    mean_value = df[column].mean()
    median_value = df[column].median()

    print(
        f"\n{column}: "
        f"Mean = {mean_value:.2f}, "
        f"Median = {median_value:.2f}, "
        f"Min = {df[column].min():.2f}, "
        f"Max = {df[column].max():.2f}"
    )

# ============================================================
# BIỂU ĐỒ 5
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)
for ax, column in zip(
    axes.flatten(),
    numeric_columns
):

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Phân phối {column}"
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Số lượng")

plt.suptitle(
    "Phân phối các biến số chính",
    fontsize=16
)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "05_numeric_distributions.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print("\nNHẬN XÉT BIỂU ĐỒ 5:")
for column in numeric_columns:
    mean_value = df[column].mean()
    median_value = df[column].median()
    if mean_value > median_value:
        print(
            f"- {column}: Mean lớn hơn Median, "
            "phân phối có xu hướng lệch phải."
        )
    elif mean_value < median_value:
        print(
            f"- {column}: Mean nhỏ hơn Median, "
            "phân phối có xu hướng lệch trái."
        )
    else:
        print(
            f"- {column}: Mean và Median gần bằng nhau."
        )

# ============================================================
# 9. TOP 10 QUỐC GIA
# ============================================================
print("\n[8] TOP 10 QUỐC GIA")
top_countries = (
    df["country"]
    .value_counts()
    .head(10)
)
print(top_countries)

# ============================================================
# BIỂU ĐỒ 6
# ============================================================
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)
plt.title(
    "Top 10 quốc gia có nhiều booking nhất",
    fontsize=14
)
plt.xlabel("Số lượng booking")
plt.ylabel("Quốc gia")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "06_top10_countries.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

top_country = top_countries.index[0]
top_country_count = top_countries.iloc[0]

top_country_percent = (
    top_country_count
    / len(df)
    * 100
)

print("\nNHẬN XÉT BIỂU ĐỒ 6:")
print(
    f"- Quốc gia có nhiều booking nhất là {top_country} "
    f"với {top_country_count:,} booking."
)

print(
    f"- Số booking của {top_country} chiếm "
    f"{top_country_percent:.2f}% tổng số booking."
)

# ============================================================
# 10. MARKET SEGMENT VÀ TỶ LỆ HỦY
# ============================================================
print("\n[9] TỶ LỆ HỦY THEO MARKET SEGMENT")
market_segment = (
    df.groupby("market_segment")
    .agg(
        booking_count=("is_canceled", "size"),
        cancellation_rate=("is_canceled", "mean")
    )
    .reset_index()
)

market_segment["cancellation_rate"] *= 100
print(market_segment)

# ============================================================
# BIỂU ĐỒ 7
# ============================================================
plt.figure(figsize=(12, 6))
sns.barplot(
    data=market_segment,
    x="market_segment",
    y="cancellation_rate"
)
plt.title(
    "Tỷ lệ hủy theo Market Segment",
    fontsize=14
)
plt.xlabel("Market Segment")
plt.ylabel("Tỷ lệ hủy (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "07_market_segment_cancellation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

market_max = market_segment.loc[market_segment["cancellation_rate"].idxmax()]
market_min = market_segment.loc[market_segment["cancellation_rate"].idxmin()]

print("\nNHẬN XÉT BIỂU ĐỒ 7:")
print(
    f"- Market Segment có tỷ lệ hủy cao nhất: "
    f"{market_max['market_segment']} "
    f"({market_max['cancellation_rate']:.2f}%), "
    f"với {market_max['booking_count']:,} booking."
)

print(
    f"- Market Segment có tỷ lệ hủy thấp nhất: "
    f"{market_min['market_segment']} "
    f"({market_min['cancellation_rate']:.2f}%), "
    f"với {market_min['booking_count']:,} booking."
)

# ============================================================
# 11. DEPOSIT TYPE VÀ TỶ LỆ HỦY
# ============================================================
print("\n[10] TỶ LỆ HỦY THEO DEPOSIT TYPE")
deposit_type = (
    df.groupby("deposit_type")
    .agg(
        booking_count=("is_canceled", "size"),
        cancellation_rate=("is_canceled", "mean")
    )
    .reset_index()
)

deposit_type["cancellation_rate"] *= 100
print(deposit_type)

# ============================================================
# BIỂU ĐỒ 8
# ============================================================
plt.figure(figsize=(10, 6))
sns.barplot(
    data=deposit_type,
    x="deposit_type",
    y="cancellation_rate"
)

plt.title(
    "Tỷ lệ hủy theo Deposit Type",
    fontsize=14
)
plt.xlabel("Deposit Type")
plt.ylabel("Tỷ lệ hủy (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "08_deposit_type_cancellation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

deposit_max = deposit_type.loc[
    deposit_type["cancellation_rate"].idxmax()
]

deposit_min = deposit_type.loc[
    deposit_type["cancellation_rate"].idxmin()
]

print("\nNHẬN XÉT BIỂU ĐỒ 8:")
print(
    f"- Deposit Type có tỷ lệ hủy cao nhất: "
    f"{deposit_max['deposit_type']} "
    f"({deposit_max['cancellation_rate']:.2f}%), "
    f"với {deposit_max['booking_count']:,} booking."
)

print(
    f"- Deposit Type có tỷ lệ hủy thấp nhất: "
    f"{deposit_min['deposit_type']} "
    f"({deposit_min['cancellation_rate']:.2f}%), "
    f"với {deposit_min['booking_count']:,} booking."
)

# ============================================================
# 12. KẾT LUẬN EDA
# ============================================================
print("\n" + "=" * 70)
print("KẾT LUẬN EDA")
print("=" * 70)
print(f"- Dataset sau cleaning có {len(df):,} booking.")
print(f"- Tỷ lệ booking bị hủy: {cancelled:.2f}%.")
print(f"- Tỷ lệ booking không bị hủy: {not_cancelled:.2f}%.")
print(f"- Tháng có nhiều booking nhất: {peak_month} ({peak_value:,} booking).")
print(f"- Tháng có ít booking nhất: {lowest_month} ({lowest_value:,} booking).")
print(f"- Quốc gia có nhiều booking nhất: {top_country} ({top_country_count:,} booking).")
print("\nCác nhận xét trên là mô tả từ dữ liệu và không khẳng định quan hệ nhân quả.")
print("\nCác biểu đồ đã được lưu tại:")
print(FIGURE_DIR)

print("=" * 70)