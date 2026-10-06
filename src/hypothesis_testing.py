"""
TV3 - Hypothesis Testing
Dataset: Hotel Booking Demand

File này chỉ thực hiện:
1. Đọc dữ liệu đã làm sạch
2. Kiểm tra các biến dùng cho kiểm định
3. Thực hiện kiểm định t-test
4. Thực hiện kiểm định Chi-square
5. Thực hiện kiểm định ANOVA
6. Thực hiện post-hoc khi cần
7. Vẽ biểu đồ hỗ trợ
8. In và lưu kết quả kiểm định

Dữ liệu đầu vào:
data/processed/hotel_bookings_cleaned.csv

Biểu đồ:
results/figures/hypothesis_testing/

Bảng kết quả:
results/tables/hypothesis_testing/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from scipy import stats
from statsmodels.stats.oneway import anova_oneway

ALPHA = 0.05


# =========================
# 1. Cấu hình đường dẫn
# =========================

DATA_PATH = "data/processed/hotel_bookings_cleaned.csv"

FIGURE_DIR = "results/figures/hypothesis_testing"
TABLE_DIR = "results/tables/hypothesis_testing"


# Tạo thư mục nếu chưa tồn tại
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# =========================
# 2. Đọc dữ liệu
# =========================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("TV3 - HYPOTHESIS TESTING")
print("=" * 60)

print("\nKích thước dữ liệu:")
print(df.shape)


# =========================
# 3. Kiểm tra các biến cần dùng
# =========================

required_columns = [
    "hotel",
    "adr",
    "is_canceled",
    "market_segment"
]

print("\nCác biến sử dụng:")
print(required_columns)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Thiếu các cột cần thiết: {missing_columns}"
    )


# Chỉ lấy các biến TV3 cần quan tâm để kiểm tra
test_data = df[required_columns]

print("\n5 dòng đầu:")
print(test_data.head())

print("\nKiểu dữ liệu:")
print(test_data.dtypes)

print("\nSố lượng giá trị thiếu:")
print(test_data.isnull().sum())


# =========================
# 4. Kiểm tra giá trị của biến phân loại
# =========================

print("\nSố lượng theo hotel:")
print(df["hotel"].value_counts(dropna=False))

print("\nSố lượng theo is_canceled:")
print(df["is_canceled"].value_counts(dropna=False))

print("\nSố lượng theo market_segment:")
print(df["market_segment"].value_counts(dropna=False))


# =========================
# 5. Kiểm tra nhanh ADR
# =========================

print("\nThống kê mô tả ADR:")
print(df["adr"].describe())


# =========================
# 6. Independent Samples t-test
# ADR giữa City Hotel và Resort Hotel
# =========================

print("\n" + "=" * 60)
print("T-TEST: ADR GIỮA CITY HOTEL VÀ RESORT HOTEL")
print("\nGiả thuyết:")
print("H0: ADR trung bình của City Hotel và Resort Hotel bằng nhau.")
print("H1: ADR trung bình của City Hotel và Resort Hotel khác nhau.")
print("=" * 60)

# Tách ADR thành 2 nhóm
city_adr = df.loc[
    df["hotel"] == "City Hotel",
    "adr"
].dropna()

resort_adr = df.loc[
    df["hotel"] == "Resort Hotel",
    "adr"
].dropna()


# -------------------------
# 6.1. Thống kê mô tả
# -------------------------

print("\nThống kê mô tả:")

print(f"City Hotel:")
print(f"  n    = {len(city_adr)}")
print(f"  Mean = {city_adr.mean():.4f}")
print(f"  Std  = {city_adr.std():.4f}")

print(f"\nResort Hotel:")
print(f"  n    = {len(resort_adr)}")
print(f"  Mean = {resort_adr.mean():.4f}")
print(f"  Std  = {resort_adr.std():.4f}")


# -------------------------
# 6.2. Levene's Test
# Kiểm tra phương sai 2 nhóm
# -------------------------

levene_stat, levene_p = stats.levene(
    city_adr,
    resort_adr
)

print("\nLevene's Test:")
print(f"Statistic = {levene_stat:.4f}")
if levene_p < 0.001:
    print("p-value   < 0.001")
else:
    print(f"p-value   = {levene_p:.6f}")


# Nếu p < 0.05:
# phương sai 2 nhóm khác nhau
# -> sử dụng Welch's t-test

equal_variance = levene_p >= ALPHA


# -------------------------
# 6.3. T-test
# -------------------------


t_stat, p_value = stats.ttest_ind(
    city_adr,
    resort_adr,
    equal_var=equal_variance
)

test_name = (
    "Student's independent t-test"
    if equal_variance
    else "Welch's independent t-test"
)

print(f"\nPhương pháp sử dụng: {test_name}")

print("\nKết quả t-test:")
print(f"t-statistic = {t_stat:.4f}")

if p_value < 0.001:
    print("p-value     < 0.001")
else:
    print(f"p-value     = {p_value:.6f}")


# -------------------------
# 6.4. Khoảng tin cậy 95% cho chênh lệch trung bình
# -------------------------

mean_diff = city_adr.mean() - resort_adr.mean()

var_city = city_adr.var(ddof=1)
var_resort = resort_adr.var(ddof=1)

n_city = len(city_adr)
n_resort = len(resort_adr)

if equal_variance:
    # Student's t-test:
    # dùng pooled variance và df = n1 + n2 - 2
    pooled_variance = (
        ((n_city - 1) * var_city)
        + ((n_resort - 1) * var_resort)
    ) / (n_city + n_resort - 2)

    se_diff = np.sqrt(
        pooled_variance
        * (1 / n_city + 1 / n_resort)
    )

    df_ci = n_city + n_resort - 2

else:
    # Welch's t-test:
    # không giả định phương sai hai nhóm bằng nhau
    se_diff = np.sqrt(
        var_city / n_city
        + var_resort / n_resort
    )

    df_ci = (
        var_city / n_city
        + var_resort / n_resort
    ) ** 2 / (
        (var_city / n_city) ** 2 / (n_city - 1)
        +
        (var_resort / n_resort) ** 2 / (n_resort - 1)
    )

critical_t = stats.t.ppf(
    1 - ALPHA / 2,
    df_ci
)

ci_lower = mean_diff - critical_t * se_diff
ci_upper = mean_diff + critical_t * se_diff

print(f"\nChênh lệch trung bình = {mean_diff:.4f}")
print(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

# -------------------------
# 6.5. Kết luận
# -------------------------


if p_value < ALPHA:

    if city_adr.mean() > resort_adr.mean():
        direction = "City Hotel có ADR trung bình cao hơn Resort Hotel"
    else:
        direction = "Resort Hotel có ADR trung bình cao hơn City Hotel"

    conclusion = (
        "Bác bỏ H0. Có sự khác biệt có ý nghĩa thống kê "
        "về ADR trung bình giữa City Hotel và Resort Hotel. "
        f"{direction}."
    )
else:
    conclusion = (
        "Chưa đủ bằng chứng để bác bỏ H0. "
        "Chưa phát hiện sự khác biệt có ý nghĩa thống kê "
        "về ADR trung bình giữa City Hotel và Resort Hotel."
    )

print("\nKết luận:")
print(conclusion)


# -------------------------
# 6.6. Cohen's d
# Đo mức độ khác biệt thực tế
# -------------------------

n1 = len(city_adr)
n2 = len(resort_adr)

std1 = city_adr.std(ddof=1)
std2 = resort_adr.std(ddof=1)

pooled_std = np.sqrt(
    (
        (n1 - 1) * std1**2
        + (n2 - 1) * std2**2
    )
    / (n1 + n2 - 2)
)

cohens_d = (
    city_adr.mean() - resort_adr.mean()
) / pooled_std

abs_d = abs(cohens_d)

if abs_d < 0.2:
    d_level = "rất nhỏ"
elif abs_d < 0.5:
    d_level = "nhỏ"
elif abs_d < 0.8:
    d_level = "trung bình"
else:
    d_level = "lớn"

print(
    f"\nCohen's d = {cohens_d:.4f} "
    f"({d_level})"
)


# -------------------------
# 6.7. Biểu đồ
# -------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="hotel",
    y="adr"
)

plt.title("Phân bố ADR theo loại khách sạn")
plt.xlabel("Loại khách sạn")
plt.ylabel("ADR")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "ttest_adr_by_hotel.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------
# 6.8. Lưu kết quả
# -------------------------

ttest_result = pd.DataFrame({
    "test": [test_name],
    "city_n": [len(city_adr)],
    "city_mean": [city_adr.mean()],
    "city_std": [city_adr.std()],
    "resort_n": [len(resort_adr)],
    "resort_mean": [resort_adr.mean()],
    "resort_std": [resort_adr.std()],
    "levene_statistic": [levene_stat],
    "levene_p_value": [levene_p],
    "t_statistic": [t_stat],
    "p_value": [p_value],
    "mean_difference": [mean_diff],
    "ci_95_lower": [ci_lower],
    "ci_95_upper": [ci_upper],
    "cohens_d": [cohens_d],
    "effect_size_level": [d_level],
    "conclusion": [conclusion]
})

ttest_result.to_csv(
    os.path.join(
        TABLE_DIR,
        "ttest_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)



# =========================
# 7. Chi-square Test of Independence
# hotel và is_canceled
# =========================

print("\n" + "=" * 60)
print("CHI-SQUARE: HOTEL VÀ IS_CANCELED")
print("\nGiả thuyết:")
print("H0: Loại khách sạn và trạng thái hủy đặt phòng độc lập.")
print("H1: Loại khách sạn và trạng thái hủy đặt phòng có mối liên hệ.")
print("=" * 60)


# -------------------------
# 7.1. Tạo bảng chéo
# -------------------------

contingency_table = pd.crosstab(
    df["hotel"],
    df["is_canceled"]
)

print("\nBảng tần số quan sát:")
print(contingency_table)


# -------------------------
# 7.2. Chi-square test
# -------------------------

chi2_stat, p_value_chi, dof, expected = stats.chi2_contingency(
    contingency_table,
    correction=False
)

print("\nKết quả Chi-square:")
print(f"Chi-square statistic = {chi2_stat:.4f}")
print(f"Degrees of freedom   = {dof}")

if p_value_chi < 0.001:
    print("p-value              < 0.001")
else:
    print(f"p-value              = {p_value_chi:.6f}")


# -------------------------
# 7.3. Expected frequencies
# -------------------------

expected_df = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

print("\nExpected frequencies:")
print(expected_df)

min_expected_freq = expected_df.min().min()

print(
    "\nExpected frequency nhỏ nhất = "
    f"{min_expected_freq:.4f}"
)

if min_expected_freq >= 5:
    print("Thỏa điều kiện Chi-square: tất cả expected frequencies >= 5.")
else:
    print(
        "Cảnh báo: có expected frequency < 5. "
        "Nên cân nhắc Fisher's Exact Test."
    )


# -------------------------
# 7.4. Cramer's V
# Đo mức độ liên hệ
# -------------------------

n = contingency_table.to_numpy().sum()

r, k = contingency_table.shape

cramers_v = np.sqrt(
    chi2_stat / (n * min(r - 1, k - 1))
)

# Diễn giải mức độ liên hệ
if cramers_v < 0.1:
    v_level = "rất yếu"
elif cramers_v < 0.3:
    v_level = "yếu"
elif cramers_v < 0.5:
    v_level = "trung bình"
else:
    v_level = "mạnh"

print(
    f"\nCramer's V = {cramers_v:.4f} "
    f"({v_level})"
)


# -------------------------
# 7.5. Tỷ lệ hủy theo loại khách sạn
# -------------------------

cancellation_rate = pd.crosstab(
    df["hotel"],
    df["is_canceled"],
    normalize="index"
) * 100

print("\nTỷ lệ theo từng loại khách sạn (%):")
print(cancellation_rate)

city_cancel_rate = cancellation_rate.loc["City Hotel", 1]
resort_cancel_rate = cancellation_rate.loc["Resort Hotel", 1]

cancel_diff = city_cancel_rate - resort_cancel_rate
abs_cancel_diff = abs(cancel_diff)

# Xác định khách sạn nào có tỷ lệ hủy cao hơn
if city_cancel_rate > resort_cancel_rate:
    direction_chi = (
        f"City Hotel có tỷ lệ hủy cao hơn Resort Hotel "
        f"({city_cancel_rate:.2f}% so với "
        f"{resort_cancel_rate:.2f}%)"
    )

elif city_cancel_rate < resort_cancel_rate:
    direction_chi = (
        f"Resort Hotel có tỷ lệ hủy cao hơn City Hotel "
        f"({resort_cancel_rate:.2f}% so với "
        f"{city_cancel_rate:.2f}%)"
    )

else:
    direction_chi = (
        f"Hai loại khách sạn có tỷ lệ hủy bằng nhau "
        f"({city_cancel_rate:.2f}%)"
    )

print("\nTỷ lệ hủy đặt phòng:")
print(f"City Hotel: {city_cancel_rate:.2f}%")
print(f"Resort Hotel: {resort_cancel_rate:.2f}%")
print(f"Chênh lệch: {abs_cancel_diff:.2f} điểm phần trăm")


# -------------------------
# 7.6. Kết luận
# -------------------------

if p_value_chi < ALPHA:
    chi_conclusion = (
        "Bác bỏ H0. Có mối liên hệ có ý nghĩa thống kê "
        "giữa loại khách sạn và trạng thái hủy đặt phòng. "
        f"{direction_chi}, "
        f"chênh lệch {abs_cancel_diff:.2f} điểm phần trăm. "
        f"Mức độ liên hệ {v_level} "
        f"(Cramer's V = {cramers_v:.4f})."
    )
else:
    chi_conclusion = (
        "Chưa đủ bằng chứng để bác bỏ H0. "
        "Chưa phát hiện mối liên hệ có ý nghĩa thống kê "
        "giữa loại khách sạn và trạng thái hủy đặt phòng."
    )

print("\nKết luận:")
print(chi_conclusion)


# -------------------------
# 7.7. Lưu kết quả Chi-square
# -------------------------

chi_result = pd.DataFrame({
    "test": ["Chi-square Test of Independence"],
    "chi2_statistic": [chi2_stat],
    "dof": [dof],
    "p_value": [p_value_chi],
    "cramers_v": [cramers_v],
    "effect_size_level": [v_level],
    "min_expected_freq": [min_expected_freq],
    "city_cancel_rate_pct": [city_cancel_rate],
    "resort_cancel_rate_pct": [resort_cancel_rate],
    "cancel_rate_difference_pct_point": [abs_cancel_diff],
    "conclusion": [chi_conclusion]
})

chi_result.to_csv(
    os.path.join(
        TABLE_DIR,
        "chi_square_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# Lưu bảng tần số quan sát
contingency_table.to_csv(
    os.path.join(
        TABLE_DIR,
        "chi_square_contingency_table.csv"
    ),
    encoding="utf-8-sig"
)


# -------------------------
# 7.8. Biểu đồ tỷ lệ hủy
# -------------------------

cancel_plot = pd.DataFrame({
    "hotel": cancellation_rate.index,
    "cancellation_rate": cancellation_rate[1].values
})

plt.figure(figsize=(8, 6))

sns.barplot(
    data=cancel_plot,
    x="hotel",
    y="cancellation_rate"
)

plt.title("Tỷ lệ hủy đặt phòng theo loại khách sạn")
plt.xlabel("Loại khách sạn")
plt.ylabel("Tỷ lệ hủy (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "chi_square_cancellation_by_hotel.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# =========================
# 8. One-way ANOVA
# ADR giữa các market_segment
# =========================

print("\n" + "=" * 60)
print("ANOVA: ADR GIỮA CÁC MARKET SEGMENT")
print("\nGiả thuyết:")
print("H0: ADR trung bình của tất cả market segment bằng nhau.")
print("H1: Có ít nhất một market segment có ADR trung bình khác.")
print("=" * 60)

# -------------------------
# 8.1. Chuẩn bị dữ liệu
# Loại các nhóm có kích thước mẫu < 30
# -------------------------

segment_counts = df["market_segment"].value_counts()

print("\nSố lượng ban đầu theo market_segment:")
print(segment_counts)

valid_segments = segment_counts[
    segment_counts >= 30
].index

excluded_segments = segment_counts[
    segment_counts < 30
]

print("\nCác nhóm bị loại vì kích thước mẫu < 30:")
print(excluded_segments)

anova_data = df[
    df["market_segment"].isin(valid_segments)
].copy()

print("\nSố lượng theo market_segment:")
print(anova_data["market_segment"].value_counts())


# -------------------------
# 8.2. Thống kê mô tả theo nhóm
# -------------------------

anova_summary = (
    anova_data
    .groupby("market_segment")["adr"]
    .agg(["count", "mean", "std", "median"])
    .sort_values("mean", ascending=False)
)

print("\nThống kê mô tả ADR theo market_segment:")
print(anova_summary)


# -------------------------
# 8.3. Tạo các nhóm ADR
# -------------------------

groups = [
    group["adr"].dropna().values
    for _, group in anova_data.groupby("market_segment")
]


# -------------------------
# 8.4. Levene's Test
# Kiểm tra đồng nhất phương sai
# -------------------------

levene_stat_anova, levene_p_anova = stats.levene(
    *groups
)

print("\nLevene's Test:")
print(f"Statistic = {levene_stat_anova:.4f}")

if levene_p_anova < 0.001:
    print("p-value   < 0.001")
else:
    print(f"p-value   = {levene_p_anova:.6f}")


# -------------------------
# 8.5. Chọn ANOVA phù hợp dựa trên Levene's Test
# -------------------------

if levene_p_anova >= ALPHA:

    print("\nPhương sai giữa các nhóm không khác biệt có ý nghĩa.")
    print("Sử dụng One-way ANOVA.")

    f_stat, p_value_anova = stats.f_oneway(
        *groups
    )

    anova_method = "One-way ANOVA"

else:

    print("\nPhương sai giữa các nhóm khác biệt có ý nghĩa.")
    print("Sử dụng Welch's ANOVA.")


    welch_result = anova_oneway(
        groups,
        use_var="unequal",
        welch_correction=True
    )

    f_stat = welch_result.statistic
    p_value_anova = welch_result.pvalue

    anova_method = "Welch's ANOVA"


print(f"\nPhương pháp sử dụng: {anova_method}")
print(f"F-statistic = {f_stat:.4f}")

if p_value_anova < 0.001:
    print("p-value     < 0.001")
else:
    print(f"p-value     = {p_value_anova:.6f}")


# -------------------------
# 8.6. Kết luận
# -------------------------


if p_value_anova < ALPHA:
    anova_conclusion = (
        "Bác bỏ H0. Có ít nhất một market segment "
        "có ADR trung bình khác biệt có ý nghĩa thống kê."
    )
else:
    anova_conclusion = (
        "Chưa đủ bằng chứng để bác bỏ H0. "
        "Chưa phát hiện sự khác biệt có ý nghĩa thống kê "
        "về ADR trung bình giữa các market segment."
    )

print("\nKết luận:")
print(anova_conclusion)


# -------------------------
# 8.7. Post-hoc Test
# Games-Howell nếu phương sai không đồng nhất
# Tukey HSD nếu phương sai đồng nhất
# -------------------------

if p_value_anova < ALPHA:
    if levene_p_anova < ALPHA:
        posthoc = pg.pairwise_gameshowell(
            data=anova_data,
            dv="adr",
            between="market_segment"
        )
        posthoc_name = "games_howell_results.csv"
        posthoc_method = "Games-Howell"
    else:
        posthoc = pg.pairwise_tukey(
            data=anova_data,
            dv="adr",
            between="market_segment"
        )
        posthoc_name = "tukey_results.csv"
        posthoc_method = "Tukey HSD"

    print(f"\n{posthoc_method} Post-hoc Test:")
    print(posthoc)

    posthoc.to_csv(
        os.path.join(TABLE_DIR, posthoc_name),
        index=False,
        encoding="utf-8-sig"
    )


# -------------------------
# 8.8. Eta-squared
# -------------------------

grand_mean = anova_data["adr"].mean()

ss_between = sum(
    len(group) * (group["adr"].mean() - grand_mean) ** 2
    for _, group in anova_data.groupby("market_segment")
)

ss_total = (
    (anova_data["adr"] - grand_mean) ** 2
).sum()

eta_squared = ss_between / ss_total

if eta_squared < 0.01:
    eta_level = "rất nhỏ"
elif eta_squared < 0.06:
    eta_level = "nhỏ"
elif eta_squared < 0.14:
    eta_level = "trung bình"
else:
    eta_level = "lớn"

print(
    f"\nEta-squared = {eta_squared:.4f} "
    f"({eta_level})"
)


# -------------------------
# 8.9. Biểu đồ
# -------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=anova_data,
    x="market_segment",
    y="adr"
)

plt.title("Phân bố ADR theo Market Segment")
plt.xlabel("Market Segment")
plt.ylabel("ADR")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "anova_adr_by_market_segment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------
# 8.10. Lưu kết quả
# -------------------------

anova_result = pd.DataFrame({
    "test": [anova_method],
    "f_statistic": [f_stat],
    "p_value": [p_value_anova],
    "levene_statistic": [levene_stat_anova],
    "levene_p_value": [levene_p_anova],
    "eta_squared": [eta_squared],
    "effect_size_level": [eta_level],
    "conclusion": [anova_conclusion]
})

anova_result.to_csv(
    os.path.join(
        TABLE_DIR,
        "anova_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

anova_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "anova_descriptive_statistics.csv"
    ),
    encoding="utf-8-sig"
)