# TV4 — CORRELATION ANALYSIS 
# Hotel Booking Demand

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình hiển thị cho Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Cấu hình giao diện đồ thị
sns.set_theme(style="whitegrid")

# 1. TẠO CÁC THƯ MỤC LƯU KẾT QUẢ
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# 2. READ DATA & PREPARATION
file_path = "data/processed/hotel_bookings_cleaned.csv" 
df = pd.read_csv(file_path)

print("Kích thước dữ liệu:", df.shape)
print("\nDanh sách các cột:")
print(df.columns.tolist())

# 3. CHỌN CÁC BIẾN SỐ 
numeric_cols = [
    'lead_time',
    'arrival_date_week_number',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'babies',
    'booking_changes',
    'previous_cancellations',
    'previous_bookings_not_canceled',
    'days_in_waiting_list',
    'adr',
    'required_car_parking_spaces',
    'total_of_special_requests',
    'total_guests',
    'total_nights'
]

# Thêm is_canceled nếu có trong dataset để phân tích yếu tố hủy phòng
if 'is_canceled' in df.columns:
    numeric_cols.append('is_canceled')

# Chỉ lấy những cột thực sự tồn tại trong dataset
numeric_cols = [col for col in numeric_cols if col in df.columns]
numeric_df = df[numeric_cols].copy()

print(f"\nDanh sách {len(numeric_cols)} biến số được chọn:")
print(numeric_cols)

# 4. KIỂM TRA MISSING VALUES
print("\n" + "=" * 60)
print("MISSING VALUES IN NUMERIC VARIABLES")
print("=" * 60)
print(numeric_df.isnull().sum())

# 5. PEARSON & SPEARMAN CORRELATION MATRIX
pearson_corr = numeric_df.corr(method='pearson')
spearman_corr = numeric_df.corr(method='spearman')

# LƯU MA TRẬN RA FILE CSV
pearson_corr.to_csv("results/tables/pearson_correlation.csv")
spearman_corr.to_csv("results/tables/spearman_correlation.csv")
print("\nĐã lưu ma trận Pearson và Spearman vào thư mục results/tables/")

# 7. PEARSON HEATMAP
plt.figure(figsize=(15, 11))
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Pearson Correlation Heatmap", fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/plots/pearson_heatmap.png", dpi=300)
plt.show()

# 8. SPEARMAN HEATMAP
plt.figure(figsize=(15, 11))
sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Spearman Correlation Heatmap", fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/plots/spearman_heatmap.png", dpi=300)
plt.show()

# 9. TÌM CÁC CẶP BIẾN CÓ TƯƠNG QUAN MẠNH NHẤT
upper_triangle = pearson_corr.where(
    np.triu(np.ones(pearson_corr.shape), k=1).astype(bool)
)

correlation_pairs = upper_triangle.stack().reset_index()
correlation_pairs.columns = ['Variable_1', 'Variable_2', 'Pearson_Correlation']
correlation_pairs['Absolute_Correlation'] = correlation_pairs['Pearson_Correlation'].abs()
correlation_pairs = correlation_pairs.sort_values(by='Absolute_Correlation', ascending=False)

# LƯU TOP 15 CẶP TƯƠNG QUAN RA CSV
top_pairs_df = correlation_pairs[['Variable_1', 'Variable_2', 'Pearson_Correlation']].head(15).round(3)
top_pairs_df.to_csv("results/tables/top_correlation_pairs.csv", index=False)

print("\n" + "=" * 60)
print("TOP 15 CORRELATION PAIRS (PEARSON)")
print("=" * 60)
print(top_pairs_df)

# 10. PHÂN TÍCH TƯƠNG QUAN VỚI TRẠNG THÁI HỦY PHÒNG (is_canceled)
if 'is_canceled' in numeric_df.columns:
    print("\n" + "=" * 60)
    print("TƯƠNG QUAN VỚI TRẠNG THÁI HỦY PHÒNG (is_canceled)")
    print("=" * 60)

    cancel_pearson = numeric_df.corr(method='pearson')['is_canceled'].drop('is_canceled')
    cancel_spearman = numeric_df.corr(method='spearman')['is_canceled'].drop('is_canceled')

    cancel_corr = pd.DataFrame({
        'Pearson': cancel_pearson,
        'Spearman': cancel_spearman
    })

    cancel_corr['Abs_Pearson'] = cancel_corr['Pearson'].abs()
    cancel_corr = cancel_corr.sort_values(by='Abs_Pearson', ascending=False)

    # LƯU BẢNG TƯƠNG QUAN HỦY PHÒNG RA CSV
    cancel_corr[['Pearson', 'Spearman']].round(3).to_csv("results/tables/cancel_correlation.csv")

    print("\nPearson và Spearman với is_canceled:")
    print(cancel_corr[['Pearson', 'Spearman']].round(3))

    # Biểu đồ Barh hủy phòng
    plt.figure(figsize=(10, 6))
    cancel_corr['Pearson'].sort_values().plot(
        kind='barh',
        color='skyblue',
        edgecolor='black'
    )
    plt.title("Tương quan của các biến với trạng thái Hủy phòng (is_canceled)", fontsize=14)
    plt.xlabel("Hệ số tương quan Pearson")
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig("results/plots/cancel_correlation_bar.png", dpi=300)
    plt.show()

# 11. SCATTER PLOTS & XU HƯỚNG MÙA VỤ
# Scatter Plot 1: Lead Time vs Total Nights
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='lead_time', y='total_nights', alpha=0.15, s=25)
sns.regplot(data=df, x='lead_time', y='total_nights', scatter=False, color='red')
plt.title("Lead Time vs Total Nights", fontsize=14)
plt.xlabel("Lead Time (days)")
plt.ylabel("Total Nights")
plt.tight_layout()
plt.savefig("results/plots/scatter_lead_time_total_nights.png", dpi=300)
plt.show()

# Scatter Plot 2: Lead Time vs Stays in Week Nights
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x='lead_time', y='stays_in_week_nights', alpha=0.15, s=20)
sns.regplot(data=df, x='lead_time', y='stays_in_week_nights', scatter=False, color='red')
plt.title("Lead Time vs Stays in Week Nights", fontsize=14)
plt.xlabel("Lead Time (days)")
plt.ylabel("Stays in Week Nights")
plt.tight_layout()
plt.savefig("results/plots/scatter_lead_time_week_nights.png", dpi=300)
plt.show()

# Line Plot: Arrival Date Week Number vs ADR
plt.figure(figsize=(11, 5))
sns.lineplot(data=df, x='arrival_date_week_number', y='adr', estimator='mean', errorbar=None, color='navy', marker='o')
plt.title("Giá phòng trung bình (ADR) theo tuần trong năm", fontsize=14)
plt.xlabel("Tuần trong năm (Arrival Date Week Number)")
plt.ylabel("ADR trung bình")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("results/plots/seasonality_adr_weekly.png", dpi=300)
plt.show()

# 12. PHÂN TÍCH VÀ ĐÁNH GIÁ MỨC ĐỘ TƯƠNG QUAN CÁC CẶP TRỌNG YẾU
def correlation_strength(r):
    r_abs = abs(r)
    if r_abs < 0.10:
        return "Rất yếu"
    elif r_abs < 0.30:
        return "Yếu"
    elif r_abs < 0.50:
        return "Vừa"
    elif r_abs < 0.70:
        return "Mạnh"
    else:
        return "Rất mạnh"

important_pairs = [
    ('lead_time', 'total_nights'),
    ('arrival_date_week_number', 'adr'),
    ('lead_time', 'stays_in_week_nights')
]

print("\n" + "=" * 60)
print("ANALYSIS OF IMPORTANT CORRELATIONS")
print("=" * 60)

for var1, var2 in important_pairs:
    if var1 in pearson_corr.columns and var2 in pearson_corr.columns:
        p_val = pearson_corr.loc[var1, var2]
        s_val = spearman_corr.loc[var1, var2]
        strength = correlation_strength(p_val)
        direction = "dương" if p_val > 0 else ("âm" if p_val < 0 else "không có")
        
        print(f"\nCặp biến: {var1} <-> {var2}")
        print(f"  - Pearson  : {p_val:.3f}")
        print(f"  - Spearman : {s_val:.3f}")
        print(f"  - Đánh giá : Tương quan {direction}, mức độ {strength}")
        # Nhận xét ý nghĩa thực tiễn
        if abs(p_val) < 0.10:
            print("  - Ý nghĩa thực tiễn: Mối liên hệ rất yếu,")
            print("    do đó khó xem đây là một mối quan hệ đáng kể trong thực tế.")

        elif p_val > 0:
            print("  - Ý nghĩa thực tiễn: Hai biến có xu hướng thay đổi cùng chiều.")
            print("    Khi một biến tăng, biến còn lại có xu hướng tăng theo.")

        else:
            print("  - Ý nghĩa thực tiễn: Hai biến có xu hướng thay đổi ngược chiều.")
            print("    Khi một biến tăng, biến còn lại có xu hướng giảm.")
# 13. LƯU Ý VỀ TƯƠNG QUAN CẤU TRÚC

print("\n" + "=" * 60)
print("LƯU Ý VỀ TƯƠNG QUAN CẤU TRÚC")
print("=" * 60)

print("""
1. total_nights được tạo từ:
   stays_in_weekend_nights + stays_in_week_nights.

2. Vì vậy, tương quan cao giữa total_nights và các biến
   thành phần có thể xuất phát trực tiếp từ cách xây dựng biến.

3. total_guests cũng được tính dựa trên số lượng khách,
   vì vậy có thể có tương quan cao với adults, children và babies.

4. Các tương quan cấu trúc vẫn được giữ trong ma trận tương quan
   để khảo sát dữ liệu, nhưng cần được diễn giải thận trọng.

5. Các hệ số tương quan chỉ cho biết mức độ liên hệ giữa các biến,
   không chứng minh rằng một biến là nguyên nhân gây ra biến còn lại.

6. Pearson và Spearman được sử dụng đồng thời để có cái nhìn
   đầy đủ hơn về mối quan hệ giữa các biến.
""")
# 14. TỔNG KẾT PHÂN TÍCH

print("\n" + "=" * 60)
print("TỔNG KẾT")
print("=" * 60)

print("""
- Pearson được sử dụng để đánh giá tương quan tuyến tính giữa các biến.

- Spearman được sử dụng để đánh giá tương quan dựa trên thứ hạng
  của dữ liệu và hỗ trợ kiểm tra các mối quan hệ đơn điệu.

- Heatmap Pearson và Spearman giúp quan sát trực quan mức độ
  tương quan giữa các biến số.

- Scatter plot và đường xu hướng giúp trực quan hóa mối quan hệ
  giữa các cặp biến tiêu biểu.

- Các mối tương quan được phân tích theo chiều hướng và mức độ,
  đồng thời được xem xét về ý nghĩa thực tiễn.

- Những biến có quan hệ cấu trúc như total_nights và total_guests
  cần được phân biệt với các mối quan hệ có ý nghĩa phân tích.

- Kết quả tương quan không nên được hiểu là bằng chứng của
  quan hệ nhân quả.
""")
