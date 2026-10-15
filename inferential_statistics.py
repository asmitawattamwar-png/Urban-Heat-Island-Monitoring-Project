"""
Member 2 / Part 5: Inferential statistics for the UHI project.

Input:
    outputs/cleaned_uhi_dataset.csv

Important data limitation:
    The supplied dataset has 500 city-like records, a Land Cover category,
    Temperature (°C), and environmental/socioeconomic variables. It has no
    date, season, year, or satellite quality-band columns. Therefore this script
    tests cross-sectional group differences, not seasonal or year-to-year change.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "cleaned_uhi_dataset.csv"
OUT = ROOT / "outputs" / "member2" / "inferential_statistics"
OUT.mkdir(parents=True, exist_ok=True)
TEMP = "Temperature (°C)"
LAND = "Land Cover"
GREEN = "Urban Greenness Ratio (%)"
ALPHA = 0.05
RNG = np.random.default_rng(42)


def cohens_d(a, b):
    """Cohen's d for independent groups using pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = (((len(a)-1)*np.var(a, ddof=1)) + ((len(b)-1)*np.var(b, ddof=1))) / (len(a)+len(b)-2)
    return (np.mean(a)-np.mean(b)) / np.sqrt(pooled_var) if pooled_var > 0 else np.nan


def mean_ci(values, confidence=0.95):
    """Two-sided t confidence interval for a sample mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    mean = values.mean()
    sem = stats.sem(values)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * sem
    return mean - margin, mean + margin


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT}\nRun data_preprocessing.py first.")
    df = pd.read_csv(INPUT)
    required = [TEMP, LAND, GREEN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from cleaned dataset: {missing}")

    df[TEMP] = pd.to_numeric(df[TEMP], errors="coerce")
    df[GREEN] = pd.to_numeric(df[GREEN], errors="coerce")
    df = df.dropna(subset=[TEMP, LAND, GREEN]).copy()
    report = []
    group_summary = []

    # 1) Confidence intervals for mean temperature in each land-cover group.
    for name, group in df.groupby(LAND, observed=False):
        vals = group[TEMP].to_numpy()
        lo, hi = mean_ci(vals)
        group_summary.append({
            "group_type": "Land Cover", "group": str(name), "n": len(vals),
            "mean_temperature_C": np.mean(vals), "sd_temperature_C": np.std(vals, ddof=1),
            "mean_CI95_lower": lo, "mean_CI95_upper": hi,
            "median_temperature_C": np.median(vals)
        })
    pd.DataFrame(group_summary).to_csv(OUT / "temperature_by_land_cover.csv", index=False)

    # 2) One-way ANOVA and Kruskal-Wallis across land-cover groups.
    groups = [g[TEMP].to_numpy() for _, g in df.groupby(LAND, observed=False) if len(g) > 0]
    labels = [str(k) for k, g in df.groupby(LAND, observed=False) if len(g) > 0]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        grand_mean = df[TEMP].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum(((df[TEMP] - grand_mean) ** 2))
        eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
        report.extend([
            {"test": "One-way ANOVA: temperature across Land Cover groups",
             "statistic": f_stat, "p_value": p_anova, "effect_size": eta_sq,
             "effect_size_name": "eta-squared", "alpha": ALPHA,
             "interpretation": "Statistically significant group difference" if p_anova < ALPHA else "No statistically significant group difference detected"},
            {"test": "Kruskal-Wallis: temperature across Land Cover groups",
             "statistic": h_stat, "p_value": p_kw, "effect_size": np.nan,
             "effect_size_name": "not calculated", "alpha": ALPHA,
             "interpretation": "Statistically significant distribution difference" if p_kw < ALPHA else "No statistically significant distribution difference detected"}
        ])

        plt.figure(figsize=(9, 5))
        df.boxplot(column=TEMP, by=LAND, grid=False)
        plt.suptitle("")
        plt.title("Temperature by Land Cover")
        plt.xlabel("Land Cover")
        plt.ylabel("Temperature (°C)")
        plt.tight_layout()
        plt.savefig(OUT / "temperature_by_land_cover_boxplot.png", dpi=180)
        plt.close()

    # 3) Compare lower and higher greenness groups split at the median.
    # The median split is an analysis choice, not a natural ecological threshold.
    threshold = df[GREEN].median()
    low = df.loc[df[GREEN] <= threshold, TEMP].to_numpy()
    high = df.loc[df[GREEN] > threshold, TEMP].to_numpy()
    if len(low) >= 2 and len(high) >= 2:
        t_stat, p_welch = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")
        u_stat, p_mw = stats.mannwhitneyu(high, low, alternative="two-sided")
        diff = high.mean() - low.mean()
        # Welch CI for difference high minus low.
        v1, v0 = np.var(high, ddof=1), np.var(low, ddof=1)
        se = np.sqrt(v1 / len(high) + v0 / len(low))
        df_welch = (v1 / len(high) + v0 / len(low))**2 / (
            ((v1 / len(high))**2 / (len(high)-1)) + ((v0 / len(low))**2 / (len(low)-1))
        ) if se > 0 else np.nan
        margin = stats.t.ppf(0.975, df_welch) * se if np.isfinite(df_welch) else np.nan
        report.extend([
            {"test": "Welch t-test: high vs low greenness temperature means",
             "statistic": t_stat, "p_value": p_welch, "effect_size": cohens_d(high, low),
             "effect_size_name": "Cohen's d (high minus low)", "alpha": ALPHA,
             "interpretation": "Statistically significant mean difference" if p_welch < ALPHA else "No statistically significant mean difference detected"},
            {"test": "Mann-Whitney U: high vs low greenness temperature distributions",
             "statistic": u_stat, "p_value": p_mw, "effect_size": np.nan,
             "effect_size_name": "not calculated", "alpha": ALPHA,
             "interpretation": "Statistically significant distribution difference" if p_mw < ALPHA else "No statistically significant distribution difference detected"}
        ])
        pd.DataFrame([{
            "greenness_median_split_percent": threshold,
            "low_group_rule": f"{GREEN} <= median", "low_n": len(low),
            "low_mean_temperature_C": low.mean(), "low_mean_CI95_lower": mean_ci(low)[0],
            "low_mean_CI95_upper": mean_ci(low)[1],
            "high_group_rule": f"{GREEN} > median", "high_n": len(high),
            "high_mean_temperature_C": high.mean(), "high_mean_CI95_lower": mean_ci(high)[0],
            "high_mean_CI95_upper": mean_ci(high)[1],
            "mean_difference_high_minus_low_C": diff,
            "difference_CI95_lower": diff - margin, "difference_CI95_upper": diff + margin,
            "cohens_d_high_minus_low": cohens_d(high, low)
        }]).to_csv(OUT / "greenness_group_comparison.csv", index=False)

        plt.figure(figsize=(7, 5))
        plt.boxplot([low, high], labels=["Lower greenness", "Higher greenness"])
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature by Median-Split Urban Greenness")
        plt.tight_layout()
        plt.savefig(OUT / "temperature_by_greenness_group.png", dpi=180)
        plt.close()

    results = pd.DataFrame(report)
    results.to_csv(OUT / "hypothesis_test_results.csv", index=False)
    (OUT / "analysis_notes.txt").write_text(
        "Inferential statistics notes\n"
        "-----------------------------\n"
        f"Input: {INPUT.name}\nRows used: {len(df)}\n"
        f"Temperature mean (C): {df[TEMP].mean():.4f}\n"
        f"Temperature SD (C): {df[TEMP].std(ddof=1):.4f}\n"
        f"Land-cover categories: {', '.join(labels)}\n"
        f"Greenness median split threshold (%): {threshold:.4f}\n"
        "No date/season/year columns exist in the supplied dataset, so temporal hypothesis tests were not performed.\n"
        "The dataset appears to contain synthetic city-style records; interpret p-values as conditional on this dataset and do not claim causal or satellite-derived conclusions.\n"
        "ANOVA assumes independent observations, approximately normal residuals within groups, and comparable variances. Kruskal-Wallis is a rank-based alternative.\n"
        "Welch's t-test does not assume equal group variances; Mann-Whitney is a rank-based alternative.\n"
        "The greenness groups use the sample median and are not official ecological classifications.\n"
    )
    print(f"Saved inferential statistics outputs to: {OUT}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
