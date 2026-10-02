"""
TV1 - EDA & Data Cleaning | Dataset: Hotel Booking Demand
Chay: python tv1_eda_cleaning.py   (can file hotel_bookings.csv cung thu muc)
Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. THU THAP & KIEM TRA DU LIEU
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "hotel_bookings.csv"
OUTPUT_FILE = BASE_DIR / "hotel_bookings_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("Kích thước:", df.shape)
df.info()
print(df.head())

# ------------------------------------------------------------
# 2. MISSING VALUES
# ------------------------------------------------------------
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nMissing values:\n", pd.DataFrame({
    "so_luong": missing,
    "ti_le_%": (missing / len(df) * 100).round(2)}))

# company: ~94% thieu -> bo cot
df = df.drop(columns=["company"])
# agent: thieu = dat truc tiep, khong qua dai ly -> dien 0
df["agent"] = df["agent"].fillna(0)
# children: chi vai dong -> dien 0
df["children"] = df["children"].fillna(0)
# country: ~0.4% -> dien 'Unknown'
df["country"] = df["country"].fillna("Unknown")

# ------------------------------------------------------------
# 3. DUPLICATES
# ------------------------------------------------------------
n_dup = df.duplicated().sum()
print(f"\nSo dong trung lap: {n_dup} ({n_dup/len(df)*100:.1f}%)")
df = df.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------
# 4. LOI LOGIC / BAN GHI BAT THUONG
# ------------------------------------------------------------
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

n_before = len(df)
df = df[df["total_guests"] > 0]      # booking khong co khach
df = df[df["total_nights"] > 0]      # booking 0 dem
df = df[df["adr"] >= 0]              # gia am
print(f"Da loai {n_before - len(df)} ban ghi bat thuong logic")

# Ep kieu
df["children"] = df["children"].astype(int)
df["agent"] = df["agent"].astype(int)
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

# Tao cot ngay den (huu ich cho cac bao cao sau)
months = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
df["arrival_month_num"] = df["arrival_date_month"].map(months)
df["arrival_date"] = pd.to_datetime(dict(
    year=df["arrival_date_year"], month=df["arrival_month_num"],
    day=df["arrival_date_day_of_month"]))

# ------------------------------------------------------------
# 5. THONG KE MO TA
# ------------------------------------------------------------
print("\nThong ke mo ta (numeric):\n", df.describe().T.round(2))
print("\nThong ke mo ta (categorical):\n",
      df.select_dtypes(exclude=["number", "datetime"]).describe().T)

# ------------------------------------------------------------
# 6. OUTLIER (IQR) - phat hien truoc, xu ly sau
# ------------------------------------------------------------
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_cols = ["lead_time", "adr", "total_nights", "total_guests",
                "days_in_waiting_list"]
rows = []
for c in outlier_cols:
    lo, hi = iqr_bounds(df[c])
    n = ((df[c] < lo) | (df[c] > hi)).sum()
    rows.append([c, round(lo, 2), round(hi, 2), n, round(n / len(df) * 100, 2)])
print("\nOutlier theo IQR:\n", pd.DataFrame(
    rows, columns=["cot", "can_duoi", "can_tren", "so_outlier", "ti_le_%"]))

fig, axes = plt.subplots(1, len(outlier_cols), figsize=(18, 4))
for ax, c in zip(axes, outlier_cols):
    sns.boxplot(y=df[c], ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.savefig("boxplot_outliers_before.png", dpi=150)
plt.close()

# Xu ly: bo gia tri phi ly (adr cuc doan), con lai winsorize ve can IQR
df = df[df["adr"] < 1000]            # adr ~5400 la loi nhap lieu
for c in ["adr", "lead_time", "total_nights"]:
    lo, hi = iqr_bounds(df[c], k=3)  # k=3: chi cat outlier cuc doan
    df[c + "_clean"] = df[c].clip(lower=max(lo, 0), upper=hi)
# Giu nguyen cot goc de TV khac co the chon dung ban nao

# ------------------------------------------------------------
# 7. PHAN TICH TONG QUAN (EDA)
# ------------------------------------------------------------
sns.set_theme(style="whitegrid")

# 7.1 Ti le huy
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["is_canceled"].value_counts().plot.pie(
    autopct="%.1f%%", labels=["Khong huy", "Huy"], ax=ax[0])
ax[0].set_ylabel("")
ax[0].set_title("Ti le huy phong")
sns.countplot(data=df, x="hotel", hue="is_canceled", ax=ax[1])
ax[1].set_title("Huy phong theo loai khach san")
plt.tight_layout(); plt.savefig("eda_cancel.png", dpi=150); plt.close()

# 7.2 Mua vu
order = list(months.keys())
plt.figure(figsize=(11, 4))
sns.countplot(data=df, x="arrival_date_month", hue="hotel", order=order)
plt.xticks(rotation=45); plt.title("So luot dat phong theo thang")
plt.tight_layout(); plt.savefig("eda_seasonality.png", dpi=150); plt.close()

# 7.3 ADR theo thang
plt.figure(figsize=(11, 4))
sns.lineplot(data=df, x="arrival_month_num", y="adr", hue="hotel", marker="o")
plt.title("ADR trung binh theo thang"); plt.xlabel("Thang")
plt.tight_layout(); plt.savefig("eda_adr_month.png", dpi=150); plt.close()

# 7.4 Phan phoi cac bien so quan trong
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["lead_time", "adr", "total_nights"]):
    sns.histplot(df[c + "_clean"], bins=40, kde=True, ax=ax)
    ax.set_title(f"Phan phoi {c}")
plt.tight_layout(); plt.savefig("eda_distributions.png", dpi=150); plt.close()

# 7.5 Top 10 quoc gia
plt.figure(figsize=(9, 4))
df["country"].value_counts().head(10).plot.bar()
plt.title("Top 10 quoc gia dat phong nhieu nhat")
plt.tight_layout(); plt.savefig("eda_country.png", dpi=150); plt.close()

# 7.6 Kenh phan phoi & loai khach
print("\nTi le huy theo market_segment:\n",
      df.groupby("market_segment")["is_canceled"].mean().sort_values().round(3))
print("\nTi le huy theo deposit_type:\n",
      df.groupby("deposit_type")["is_canceled"].mean().round(3))

# ------------------------------------------------------------
# 8. XUAT DATA SACH CHO CAC THANH VIEN KHAC
# ------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print("\nDa luu hotel_bookings_cleaned.csv:", df.shape)