
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. CẤU HÌNH ĐƯỜNG DẪN & THƯ MỤC
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "hotel_bookings_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Phong cách đồ họa khoa học
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

ALPHA = 0.05
TARGET_VARS = ["adr", "lead_time", "total_of_special_requests"]


# 2. DATA VALIDATION
def load_and_validate_data(file_path):
    """Đọc dữ liệu, kiểm tra tính hợp lệ và in bảng data validation."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {file_path}")

    df = pd.read_csv(file_path)
    init_rows, init_cols = df.shape
    if init_rows == 0:
        raise ValueError("File dữ liệu rỗng (0 dòng).")

    missing_vars = [c for c in TARGET_VARS if c not in df.columns]
    if missing_vars:
        raise KeyError(f"Thiếu các cột cần thiết trong dataset: {missing_vars}")

    # Duplicate được kiểm tra để báo cáo; việc loại bỏ duplicate đã thực hiện ở bước EDA chung (TV1)
    total_missing_target = int(df[TARGET_VARS].isna().sum().sum())
    dup_count = int(df.duplicated().sum())
    valid_mask = (
        df["adr"].notna() & df["lead_time"].notna() & df["total_of_special_requests"].notna()
        & (df["adr"] >= 0) & (df["lead_time"] >= 0) & (df["total_of_special_requests"] >= 0)
    )
    clean_df = df[valid_mask].copy()

    print("DATA VALIDATION")
    print(f"Rows:                   {init_rows:,}")
    print(f"Columns:                {init_cols}")
    print(f"Missing (3 biến chính): {total_missing_target}")
    print(f"Duplicates:             {dup_count:,}")
    print(f"Rows used for analysis: {len(clean_df):,}")

    return clean_df


# 3. HÀM TIỆN ÍCH THỐNG KÊ & VẼ HÌNH
def calculate_descriptive_stats(series, var_name):
    """Tính 12 chỉ số thống kê mô tả kèm IQR bounds và số lượng outlier tiềm năng."""
    s = series.dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((s < lower_bound) | (s > upper_bound)).sum())

    return {
        "Variable": var_name,
        "Count": int(len(s)),
        "Mean": float(s.mean()),
        "Median": float(s.median()),
        "Std": float(s.std()),
        "Variance": float(s.var()),
        "Min": float(s.min()),
        "Q1": q1,
        "Q3": q3,
        "Max": float(s.max()),
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Potential_Outliers": outliers,
        "Skewness": float(s.skew()),
        "Kurtosis": float(s.kurtosis()),
    }


def plot_qq(ax, series, dist="norm", color="#2b5c8f", title="Q-Q Plot"):
    """Vẽ Q-Q plot và đường chuẩn lý thuyết."""
    (osm, osr), (slope, intercept, _) = stats.probplot(series.dropna(), dist=dist)
    ax.scatter(osm, osr, color=color, alpha=0.25, s=6, edgecolors="none", label="Quan sát")
    ax.plot(osm, slope * osm + intercept, color="#d9534f", linewidth=1.8, label="Đường chuẩn lý thuyết")
    ax.set_title(title, pad=8)
    ax.set_xlabel("Phân vị lý thuyết")
    ax.set_ylabel("Phân vị mẫu")
    ax.legend(loc="lower right")


# 4. PHÂN TÍCH BIẾN 1: ADR
def analyze_adr(df):
    """Phân tích biến liên tục ADR: Normal, Log-transform và đánh giá Log-Normal."""
    s_adr = df["adr"].dropna()
    desc = calculate_descriptive_stats(s_adr, "adr")

    # Kiểm định Normal trên ADR gốc
    k2_adr, p_k2_adr = stats.normaltest(s_adr)
    dec_norm = "Reject H0" if p_k2_adr < ALPHA else "Fail to reject H0"
    adr_norm_assessment = "Có bằng chứng dữ liệu không tuân theo Normal" if p_k2_adr < ALPHA else "Chưa có đủ bằng chứng bác bỏ Normal"

    # Log-transform đối với các giá trị dương ADR > 0
    adr_pos = s_adr[s_adr > 0]
    log_adr = np.log(adr_pos)
    k2_log, p_k2_log = stats.normaltest(log_adr)
    dec_log = "Reject H0" if p_k2_log < ALPHA else "Fail to reject H0"
    skew_log = float(log_adr.skew())

    if p_k2_log < ALPHA:
        assess_log = "Không tuân theo Normal; log-transform giảm độ lệch nhưng chưa đủ bằng chứng xác nhận Log-Normal"
        stat_conclusion = (
            f"D'Agostino K² trên log(ADR) có p = {p_k2_log:.4g} < {ALPHA} ({dec_log}). "
            f"Log-transform giúp giảm skewness từ {desc['Skewness']:.2f} xuống {skew_log:.2f}, "
            "tuy nhiên bằng chứng thống kê chưa đủ để xác nhận ADR tuân theo Log-Normal."
        )
    else:
        assess_log = "Phù hợp với giả định Log-Normal trong phạm vi mẫu"
        stat_conclusion = (
            f"D'Agostino K² trên log(ADR) có p = {p_k2_log:.4g} >= {ALPHA} ({dec_log}). "
            "Không bác bỏ H0; ADR có thể xem là phù hợp với giả định Log-Normal trong phạm vi mẫu dữ liệu."
        )

    mean_med_rel = "lớn hơn median, cho thấy dấu hiệu lệch phải" if desc["Mean"] > desc["Median"] else "nhỏ hơn median, cho thấy dấu hiệu lệch trái"
    skew_rel = "giảm rõ rệt độ lệch" if abs(skew_log) < abs(desc["Skewness"]) else "không làm giảm độ lệch"
    norm_comment = "kiểm định bác bỏ giả thuyết Normality" if p_k2_adr < ALPHA else "kiểm định chưa đủ bằng chứng bác bỏ giả thuyết Normality"

    basic_comment = (
        f"ADR có mean ({desc['Mean']:.2f}) {mean_med_rel} (Skewness = {desc['Skewness']:.2f}). "
        f"Theo quy tắc IQR, có {desc['Potential_Outliers']:,} giá trị có khả năng là ngoại lai. "
        f"Q-Q plot được sử dụng để đánh giá mức độ lệch khỏi Normal; {norm_comment} (p = {p_k2_adr:.4g}). "
        f"Phép biến đổi log(ADR) giúp {skew_rel} (Skewness sau log = {skew_log:.2f}), "
        f"tuy nhiên kiểm định normality trên log(ADR) {dec_log.lower()} (p = {p_k2_log:.4g})."
    )

    # Biểu đồ ADR
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Hình 1: Phân tích Phân phối Xác suất Biến adr (Average Daily Rate)", fontsize=13, fontweight="bold", y=0.98)

    sns.histplot(s_adr, bins=40, kde=True, color="#2b5c8f", ax=axes[0, 0], stat="density", alpha=0.6)
    axes[0, 0].axvline(desc["Mean"], color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean: {desc['Mean']:.1f}")
    axes[0, 0].axvline(desc["Median"], color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Median: {desc['Median']:.1f}")
    axes[0, 0].set_title("A. Histogram & KDE (ADR gốc)", pad=8)
    axes[0, 0].set_xlabel("ADR (EUR)")
    axes[0, 0].set_ylabel("Mật độ xác suất")
    axes[0, 0].legend()

    sns.boxplot(x=s_adr, color="#6baed6", ax=axes[0, 1], flierprops={"marker": "o", "markersize": 3, "alpha": 0.3})
    axes[0, 1].set_title(f"B. Boxplot (IQR = {desc['IQR']:.1f})", pad=8)
    axes[0, 1].set_xlabel("ADR (EUR)")

    plot_qq(axes[0, 2], s_adr, dist="norm", color="#2b5c8f", title="C. Q-Q Plot so với Normal (ADR gốc)")

    sns.histplot(log_adr, bins=40, kde=True, color="#238b45", ax=axes[1, 0], stat="density", alpha=0.6)
    x_axis = np.linspace(log_adr.min(), log_adr.max(), 200)
    axes[1, 0].plot(x_axis, stats.norm.pdf(x_axis, log_adr.mean(), log_adr.std()), color="#d9534f", linestyle="--", linewidth=2, label="Fit Normal")
    axes[1, 0].set_title(f"D. log(ADR) Histogram & KDE (Skew = {skew_log:.2f})", pad=8)
    axes[1, 0].set_xlabel("log(ADR)")
    axes[1, 0].set_ylabel("Mật độ xác suất")
    axes[1, 0].legend()

    plot_qq(axes[1, 1], log_adr, dist="norm", color="#238b45", title="E. Q-Q Plot so với Normal của log(ADR)")

    axes[1, 2].axis("off")
    summary_text = (
        f"TÓM TẮT THỐNG KÊ ADR:\n"
        f"• Quan sát: {desc['Count']:,}\n"
        f"• Mean: {desc['Mean']:.2f} | Median: {desc['Median']:.2f}\n"
        f"• Std: {desc['Std']:.2f} | IQR: {desc['IQR']:.2f}\n"
        f"• Ngưỡng ngoại lai: [{desc['Lower_Bound']:.1f}, {desc['Upper_Bound']:.1f}]\n"
        f"• Điểm ngoại lai tiềm năng: {desc['Potential_Outliers']:,}\n"
        f"• Skewness gốc: {desc['Skewness']:.2f} -> log(ADR): {skew_log:.2f}\n\n"
        f"KIỂM ĐỊNH NORMAL (D'Agostino K²):\n"
        f"• ADR gốc: K² = {k2_adr:.2f}, p = {p_k2_adr:.4g} -> {dec_norm}\n"
        f"• log(ADR): K² = {k2_log:.2f}, p = {p_k2_log:.4g} -> {dec_log}\n\n"
        f"ĐÁNH GIÁ:\n{assess_log}"
    )
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=9.2, va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_DIR / "dist_01_adr_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    test_rows = [
        {"Variable": "adr", "Distribution": "Normal", "Test": "D'Agostino's K² Test", "Statistic": float(k2_adr), "P_Value": float(p_k2_adr), "Alpha": ALPHA, "Decision": dec_norm, "Conclusion": f"K² = {k2_adr:.2f}, p = {p_k2_adr:.4g}. {dec_norm} phân phối chuẩn."},
        {"Variable": "adr", "Distribution": "Log-Normal", "Test": "D'Agostino's K² Test on log(ADR)", "Statistic": float(k2_log), "P_Value": float(p_k2_log), "Alpha": ALPHA, "Decision": dec_log, "Conclusion": stat_conclusion}
    ]

    interp_rows = [
        {"Variable": "adr", "Data_Type": "Continuous numerical", "Count": desc["Count"], "Mean": desc["Mean"], "Median": desc["Median"], "Variance": desc["Variance"], "Std": desc["Std"], "Skewness": desc["Skewness"], "Kurtosis": desc["Kurtosis"], "Distribution_Candidate": "Normal Distribution", "Test_Name": "D'Agostino's K² Test", "Test_Statistic": float(k2_adr), "P_Value": float(p_k2_adr), "Decision": dec_norm, "Distribution_Assessment": adr_norm_assessment, "Basic_Comment": basic_comment},
        {"Variable": "adr (log-transformed)", "Data_Type": "Continuous numerical", "Count": int(len(adr_pos)), "Mean": float(log_adr.mean()), "Median": float(log_adr.median()), "Variance": float(log_adr.var()), "Std": float(log_adr.std()), "Skewness": skew_log, "Kurtosis": float(log_adr.kurtosis()), "Distribution_Candidate": "Log-Normal Distribution", "Test_Name": "D'Agostino's K² Test on log(ADR)", "Test_Statistic": float(k2_log), "P_Value": float(p_k2_log), "Decision": dec_log, "Distribution_Assessment": assess_log, "Basic_Comment": basic_comment}
    ]

    summary_info = {
        "mean": desc["Mean"], "median": desc["Median"], "skewness": desc["Skewness"],
        "norm_p": p_k2_adr, "log_p": p_k2_log, "assessment": assess_log, "conclusion": stat_conclusion
    }

    return desc, test_rows, interp_rows, summary_info


# 5. PHÂN TÍCH BIẾN 2: LEAD_TIME
def analyze_lead_time(df):
    """Phân tích biến duration lead_time: kiểm tra độ lệch và đánh giá Exponential."""
    s_lt = df["lead_time"].dropna()
    desc = calculate_descriptive_stats(s_lt, "lead_time")

    scale_hat = desc["Mean"]
    lambda_hat = 1.0 / scale_hat if scale_hat > 0 else 0.0

    # KS-test với scale ước lượng từ mẫu
    ks_stat, ks_p = stats.kstest(s_lt, "expon", args=(0, scale_hat))
    dec_ks = "Reject H0" if ks_p < ALPHA else "Fail to reject H0"

    if ks_p < ALPHA:
        assess_lt = "Không phù hợp với Exponential lý thuyết (dù có dạng suy giảm)"
        stat_conclusion = (
            f"Reject H0: KS statistic = {ks_stat:.4f}, p = {ks_p:.4g} < {ALPHA}. "
            "Có bằng chứng dữ liệu không phù hợp với phân phối Exponential lý thuyết. "
            "(Tham số được ước lượng từ mẫu, do đó kết quả KS test được sử dụng như một chỉ báo đánh giá độ phù hợp hơn là bằng chứng tuyệt đối; "
            "sự không phù hợp xuất phát từ các đỉnh chu kỳ đặt phòng)."
        )
    else:
        assess_lt = "Chưa đủ bằng chứng bác bỏ Exponential"
        stat_conclusion = (
            f"Fail to reject H0: KS statistic = {ks_stat:.4f}, p = {ks_p:.4g} >= {ALPHA}. "
            "Chưa có đủ bằng chứng cho thấy dữ liệu không phù hợp với Exponential. "
            "Đây không phải là bằng chứng chứng minh dữ liệu chắc chắn tuân theo Exponential."
        )

    mean_med_rel = "lớn hơn median rõ rệt, cho thấy đuôi dài kéo sang phải" if desc["Mean"] > desc["Median"] else "nhỏ hơn median"
    basic_comment = (
        f"lead_time có mean ({desc['Mean']:.1f} ngày) {mean_med_rel} (Skewness = {desc['Skewness']:.2f}). "
        f"Khoảng IQR = {desc['IQR']:.1f} ngày với {desc['Potential_Outliers']:,} giá trị có khả năng là ngoại lai theo IQR. "
        f"Q-Q plot uốn cong xa đường Normal. "
        f"KS test đánh giá phân phối Exponential cho kết quả: {dec_ks} (D = {ks_stat:.4f}, p = {ks_p:.4g})."
    )

    # Biểu đồ lead_time
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Hình 2: Phân tích Phân phối Xác suất Biến lead_time", fontsize=13, fontweight="bold", y=0.98)

    sns.histplot(s_lt, bins=45, kde=True, color="#e6550d", ax=axes[0], stat="density", alpha=0.5, label="Thực tế (KDE)")
    x_axis = np.linspace(0, s_lt.max(), 300)
    axes[0].plot(x_axis, stats.expon.pdf(x_axis, scale=scale_hat), color="#3182bd", linewidth=2, label=f"Exponential fit (λ={lambda_hat:.4f})")
    axes[0].axvline(desc["Mean"], color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean: {desc['Mean']:.1f}")
    axes[0].axvline(desc["Median"], color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Median: {desc['Median']:.1f}")
    axes[0].set_title(f"A. Histogram, KDE & Exponential Fit (λ={lambda_hat:.4f})", pad=8)
    axes[0].set_xlabel("lead_time (ngày)")
    axes[0].set_ylabel("Mật độ xác suất")
    axes[0].legend()

    sns.boxplot(x=s_lt, color="#fdae6b", ax=axes[1], flierprops={"marker": "o", "markersize": 3, "alpha": 0.3})
    axes[1].set_title(f"B. Boxplot (IQR = {desc['IQR']:.1f})", pad=8)
    axes[1].set_xlabel("lead_time (ngày)")

    plot_qq(axes[2], s_lt, dist="norm", color="#e6550d", title="C. Q-Q Plot so với Normal")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "dist_02_lead_time_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    test_rows = [
        {"Variable": "lead_time", "Distribution": "Exponential", "Test": "Kolmogorov-Smirnov Test (Estimated Scale)", "Statistic": float(ks_stat), "P_Value": float(ks_p), "Alpha": ALPHA, "Decision": dec_ks, "Conclusion": stat_conclusion}
    ]

    interp_rows = [
        {"Variable": "lead_time", "Data_Type": "Duration / count-like", "Count": desc["Count"], "Mean": desc["Mean"], "Median": desc["Median"], "Variance": desc["Variance"], "Std": desc["Std"], "Skewness": desc["Skewness"], "Kurtosis": desc["Kurtosis"], "Distribution_Candidate": "Exponential Distribution", "Test_Name": "Kolmogorov-Smirnov Test", "Test_Statistic": float(ks_stat), "P_Value": float(ks_p), "Decision": dec_ks, "Distribution_Assessment": assess_lt, "Basic_Comment": basic_comment}
    ]

    summary_info = {
        "mean": desc["Mean"], "median": desc["Median"], "skewness": desc["Skewness"],
        "lambda": lambda_hat, "ks_stat": ks_stat, "ks_p": ks_p, "assessment": assess_lt, "conclusion": stat_conclusion
    }

    return desc, test_rows, interp_rows, summary_info


# 6. PHÂN TÍCH BIẾN 3: TOTAL_OF_SPECIAL_REQUESTS
def analyze_special_requests(df):
    """Phân tích biến đếm rời rạc: Mean-Variance, tail binning, kiểm tra E_i >= 5 và Chi-Square GOF."""
    s_sr = df["total_of_special_requests"].dropna()
    desc = calculate_descriptive_stats(s_sr, "total_of_special_requests")
    n_total = desc["Count"]
    lam = desc["Mean"]

    obs_counts = s_sr.value_counts().sort_index()

    # Gom nhóm tail: 0, 1, 2, 3 và >= 4
    bins = [0, 1, 2, 3]
    obs_binned = [int(obs_counts.get(k, 0)) for k in bins]
    obs_binned.append(int(obs_counts[obs_counts.index >= 4].sum()))

    # Tần số kỳ vọng Poisson
    exp_probs = [float(stats.poisson.pmf(k, lam)) for k in bins]
    exp_probs.append(float(1.0 - stats.poisson.cdf(3, lam)))
    exp_binned = [p * n_total for p in exp_probs]

    # Kiểm tra điều kiện tần số kỳ vọng E_i >= 5
    min_exp = min(exp_binned)
    if min_exp < 5.0:
        raise ValueError(
            f"Điều kiện kiểm định Chi-Square không thỏa mãn: Có tần số kỳ vọng E_min = {min_exp:.2f} < 5. "
            "Cần gộp thêm các nhóm đuôi để đảm bảo tính hợp lệ của kiểm định."
        )

    # Kiểm tra tổng tương thích
    sum_obs, sum_exp = sum(obs_binned), sum(exp_binned)
    if abs(sum_obs - sum_exp) > 1e-4:
        raise ValueError(f"Tổng tần số không khớp: Observed={sum_obs}, Expected={sum_exp}")

    # Chi-square Goodness-of-Fit với ddof=1 (do 1 tham số lambda được ước lượng từ dữ liệu)
    chi2_res = stats.chisquare(f_obs=obs_binned, f_exp=exp_binned, ddof=1)
    chi2_stat, chi2_p = float(chi2_res.statistic), float(chi2_res.pvalue)
    dec_chi2 = "Reject H0" if chi2_p < ALPHA else "Fail to reject H0"

    disp_ratio = desc["Variance"] / desc["Mean"] if desc["Mean"] > 0 else 1.0

    if chi2_p < ALPHA:
        assess_sr = "Không phù hợp hoàn toàn với Poisson (dù Mean gần Variance)"
        stat_conclusion = (
            f"Reject H0: Chi² = {chi2_stat:.2f}, p = {chi2_p:.4g} < {ALPHA}. "
            f"Mặc dù Mean ({desc['Mean']:.4f}) xấp xỉ Variance ({desc['Variance']:.4f}) với tỷ lệ Var/Mean = {disp_ratio:.4f} "
            "(dấu hiệu phù hợp về moment), kiểm định Chi-square bác bỏ H0. "
            "Dữ liệu có bằng chứng không phù hợp hoàn toàn với Poisson."
        )
    else:
        assess_sr = "Chưa đủ bằng chứng bác bỏ Poisson"
        stat_conclusion = (
            f"Fail to reject H0: Chi² = {chi2_stat:.2f}, p = {chi2_p:.4g} >= {ALPHA}. "
            "Chưa có đủ bằng chứng cho thấy dữ liệu không phù hợp với Poisson."
        )

    basic_comment = (
        f"total_of_special_requests là biến đếm rời rạc. "
        f"Mean ({desc['Mean']:.4f}) và Variance ({desc['Variance']:.4f}) rất gần nhau (Var/Mean = {disp_ratio:.4f}), "
        f"đây là dấu hiệu phù hợp với Poisson nhưng không phải bằng chứng chứng minh. "
        f"Chi-square Goodness-of-Fit (ddof=1) là cơ sở chính đánh giá: {dec_chi2} (Chi² = {chi2_stat:.2f}, p = {chi2_p:.4g})."
    )

    # Biểu đồ special requests
    labels = ["0", "1", "2", "3", ">= 4"]
    emp_props = [o / n_total for o in obs_binned]
    x_pos = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hình 3: Phân tích Phân phối Biến Đếm total_of_special_requests", fontsize=13, fontweight="bold", y=0.98)

    axes[0].bar(x_pos - width / 2, emp_props, width=width, label="Thực tế (Empirical PMF)", color="#3182bd", alpha=0.8, edgecolor="black")
    axes[0].bar(x_pos + width / 2, exp_probs, width=width, label=f"Lý thuyết (Poisson λ={lam:.3f})", color="#e6550d", alpha=0.8, edgecolor="black")
    axes[0].set_title(f"A. So sánh PMF Thực tế vs Poisson (Mean={lam:.3f}, Var={desc['Variance']:.3f})", pad=8)
    axes[0].set_xlabel("Số lượng yêu cầu đặc biệt")
    axes[0].set_ylabel("Tỷ lệ xác suất")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    for i in x_pos:
        axes[0].text(i - width / 2, emp_props[i] + 0.01, f"{emp_props[i]:.2f}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, exp_probs[i] + 0.01, f"{exp_probs[i]:.2f}", ha="center", fontsize=8, color="#a63603")

    diff_pct = [(emp - exp) * 100 for emp, exp in zip(emp_props, exp_probs)]
    bar_colors = ["#d9534f" if v < 0 else "#2ca02c" for v in diff_pct]
    axes[1].bar(x_pos, diff_pct, color=bar_colors, edgecolor="black", width=0.45, alpha=0.8)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_title("B. Chênh lệch Tỷ lệ (Empirical % - Poisson %)", pad=8)
    axes[1].set_xlabel("Số lượng yêu cầu đặc biệt")
    axes[1].set_ylabel("Điểm phần trăm chênh lệch (% diff)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)

    for i, v in enumerate(diff_pct):
        axes[1].text(i, v + (0.25 if v >= 0 else -0.55), f"{v:+.2f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIGURES_DIR / "dist_03_special_requests_poisson.png", dpi=300, bbox_inches="tight")
    plt.close()

    test_rows = [
        {"Variable": "total_of_special_requests", "Distribution": f"Poisson(λ={lam:.4f})", "Test": "Chi-Square Goodness-of-Fit (ddof=1, Tail >= 4)", "Statistic": float(chi2_stat), "P_Value": float(chi2_p), "Alpha": ALPHA, "Decision": dec_chi2, "Conclusion": stat_conclusion}
    ]

    interp_rows = [
        {"Variable": "total_of_special_requests", "Data_Type": "Discrete count", "Count": desc["Count"], "Mean": desc["Mean"], "Median": desc["Median"], "Variance": desc["Variance"], "Std": desc["Std"], "Skewness": desc["Skewness"], "Kurtosis": desc["Kurtosis"], "Distribution_Candidate": f"Poisson(λ={lam:.4f})", "Test_Name": "Chi-Square Goodness-of-Fit (Tail >= 4)", "Test_Statistic": float(chi2_stat), "P_Value": float(chi2_p), "Decision": dec_chi2, "Distribution_Assessment": assess_sr, "Basic_Comment": basic_comment}
    ]

    summary_info = {
        "mean": desc["Mean"], "variance": desc["Variance"], "lambda": lam,
        "chi2_stat": chi2_stat, "chi2_p": chi2_p, "assessment": assess_sr, "conclusion": stat_conclusion
    }

    return desc, test_rows, interp_rows, summary_info


# 7. XUẤT CSV
def save_results(desc_list, test_rows, interp_rows):
    """Lưu 3 file CSV chuẩn hóa vào results/tables/."""
    pd.DataFrame(desc_list).to_csv(TABLES_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(TABLES_DIR / "distribution_test_results.csv", index=False, encoding="utf-8-sig")

    interp_cols = [
        "Variable", "Data_Type", "Count", "Mean", "Median", "Variance", "Std",
        "Skewness", "Kurtosis", "Distribution_Candidate", "Test_Name",
        "Test_Statistic", "P_Value", "Decision", "Distribution_Assessment", "Basic_Comment"
    ]
    pd.DataFrame(interp_rows)[interp_cols].to_csv(TABLES_DIR / "distribution_interpretation.csv", index=False, encoding="utf-8-sig")

    print("\nĐã lưu các bảng kết quả CSV:")
    print(f"1. {TABLES_DIR / 'descriptive_statistics.csv'}")
    print(f"2. {TABLES_DIR / 'distribution_test_results.csv'}")
    print(f"3. {TABLES_DIR / 'distribution_interpretation.csv'}")


# 8. TERMINAL SUMMARY
def print_summary(adr_s, lt_s, sr_s):
    """In phần SUMMARY chuẩn mực ra terminal ở cuối chương trình."""
    print()
    print("PROBABILITY DISTRIBUTION SUMMARY")
    print()
    print("ADR")
    print(f"- Mean: {adr_s['mean']:.4g}")
    print(f"- Median: {adr_s['median']:.4g}")
    print(f"- Skewness: {adr_s['skewness']:.4g}")
    print(f"- Normality p-value: {adr_s['norm_p']:.4g}")
    print(f"- Log(ADR) normality p-value: {adr_s['log_p']:.4g}")
    print(f"- Assessment: {adr_s['assessment']}")
    print(f"- Conclusion: {adr_s['conclusion']}")
    print()

    print("LEAD_TIME")
    print(f"- Mean: {lt_s['mean']:.4g}")
    print(f"- Median: {lt_s['median']:.4g}")
    print(f"- Skewness: {lt_s['skewness']:.4g}")
    print(f"- Exponential lambda: {lt_s['lambda']:.4g}")
    print(f"- KS statistic: {lt_s['ks_stat']:.4g}")
    print(f"- KS p-value: {lt_s['ks_p']:.4g}")
    print(f"- Assessment: {lt_s['assessment']}")
    print(f"- Conclusion: {lt_s['conclusion']}")
    print()

    print("TOTAL_OF_SPECIAL_REQUESTS")
    print(f"- Mean: {sr_s['mean']:.4g}")
    print(f"- Variance: {sr_s['variance']:.4g}")
    print(f"- Lambda: {sr_s['lambda']:.4g}")
    print(f"- Chi-square statistic: {sr_s['chi2_stat']:.4g}")
    print(f"- p-value: {sr_s['chi2_p']:.4g}")
    print(f"- Assessment: {sr_s['assessment']}")
    print(f"- Conclusion: {sr_s['conclusion']}")
    print()


# 9. MAIN
def main():
    print("TV2 - PHÂN TÍCH PHÂN PHỐI XÁC SUẤT")
    clean_df = load_and_validate_data(DATA_PATH)

    all_desc, all_tests, all_interps = [], [], []

    print("\n[1/3] Đang phân tích biến: adr...")
    d_adr, t_adr, i_adr, s_adr = analyze_adr(clean_df)
    all_desc.append(d_adr); all_tests.extend(t_adr); all_interps.extend(i_adr)

    print("[2/3] Đang phân tích biến: lead_time...")
    d_lt, t_lt, i_lt, s_lt = analyze_lead_time(clean_df)
    all_desc.append(d_lt); all_tests.extend(t_lt); all_interps.extend(i_lt)

    print("[3/3] Đang phân tích biến: total_of_special_requests...")
    d_sr, t_sr, i_sr, s_sr = analyze_special_requests(clean_df)
    all_desc.append(d_sr); all_tests.extend(t_sr); all_interps.extend(i_sr)

    save_results(all_desc, all_tests, all_interps)
    print_summary(s_adr, s_lt, s_sr)
    print("Hoàn tất thành công phân tích Probability Distribution!")


if __name__ == "__main__":
    main()
