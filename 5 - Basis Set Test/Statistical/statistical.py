import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel
from itertools import combinations
from statsmodels.stats.anova import AnovaRM
import os

"""
Do basis sets produce statistically equivalent descriptor values? 
If not, what is the structure of the deviation? 
For each individual system (System, Ring), how do basis sets differ?
"""

# -----------------------------
# 1. LOAD DATA
# -----------------------------
file_path = "norm.csv"

df = pd.read_csv(file_path, sep="\t")

# Clean column names (avoid hidden spaces issues)
df.columns = df.columns.str.strip()

# Fix column name for ANOVA compatibility
df = df.rename(columns={"Basis Set": "Basis_Set"})

print("Columns found:")
print(df.columns.tolist())

# -----------------------------
# 2. CONFIG
# -----------------------------
descriptor_cols = [
    "|Q2|_(ring atoms)",
    "Q2_(zz,ring atoms)",
    "|Q2|(0)",
    "Q2(0)zz",
    "|Q2|(1)",
    "Q2(1)zz",
    "|Q2|(-1)",
    "Q2(-1)zz"
]

required_cols = ["System", "Ring", "Basis_Set"]

# Validate columns
for col in required_cols + descriptor_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Prepare log file name
base_name = os.path.splitext(os.path.basename(file_path))[0]
log_file = f"{base_name}_statistical.log"

# To store descriptor-level summary
summary_results = {}

# -----------------------------
# 3. ANALYSIS FUNCTION
# -----------------------------
def analyze_descriptor(df, descriptor, log_handle):

    log_handle.write("\n" + "="*60 + "\n")
    log_handle.write(f"Descriptor: {descriptor}\n")
    log_handle.write("="*60 + "\n")

    # Pivot table
    pivot = df.pivot_table(
        index=["System", "Ring"],
        columns="Basis_Set",
        values=descriptor
    )

    # Drop incomplete cases
    pivot = pivot.dropna()

    log_handle.write(f"\nValid paired samples: {len(pivot)}\n")

    basis_sets = pivot.columns.tolist()
    conclusions = []
    sig_overall = False  # Track if descriptor shows significance anywhere

    # -------------------------
    # 1. Friedman Test
    # -------------------------
    data = [pivot[col].values for col in basis_sets]

    stat, p = friedmanchisquare(*data)

    log_handle.write("\nFriedman test:\n")
    log_handle.write(f"Statistic = {stat:.4f}\n")
    log_handle.write(f"p-value   = {p:.4e}\n")

    if p < 0.05:
        conclusions.append(f"Friedman test: Significant differences among basis sets (p={p:.4e})")
    else:
        conclusions.append(f"Friedman test: No significant differences among basis sets (p={p:.4e})")

    # -------------------------
    # 2. Repeated Measures ANOVA
    # -------------------------
    long_df = pivot.reset_index().melt(
        id_vars=["System", "Ring"],
        var_name="Basis_Set",
        value_name="value"
    )

    long_df["subject"] = (
        long_df["System"].astype(str) + "_" +
        long_df["Ring"].astype(str)
    )

    long_df["Basis_Set"] = long_df["Basis_Set"].astype("category")

    try:
        aov = AnovaRM(
            long_df,
            depvar="value",
            subject="subject",
            within=["Basis_Set"]
        ).fit()

        log_handle.write("\nRepeated Measures ANOVA:\n")
        log_handle.write(str(aov.summary()) + "\n")

        # Effect size (eta squared)
        try:
            F = aov.anova_table["F Value"].iloc[0]
            df_num = aov.anova_table["Num DF"].iloc[0]
            df_den = aov.anova_table["Den DF"].iloc[0]

            eta_sq = (F * df_num) / (F * df_num + df_den)
            conclusions.append(f"Repeated Measures ANOVA: Effect size eta^2 = {eta_sq:.4f}")
            if aov.anova_table["Pr > F"].iloc[0] < 0.05:
                conclusions.append(f"Repeated Measures ANOVA: Significant differences detected (p={aov.anova_table['Pr > F'].iloc[0]:.4e})")
            else:
                conclusions.append(f"Repeated Measures ANOVA: No significant differences (p={aov.anova_table['Pr > F'].iloc[0]:.4e})")
        except Exception:
            pass

    except Exception as e:
        log_handle.write("\nANOVA failed: " + str(e) + "\n")
        conclusions.append("Repeated Measures ANOVA: Failed to compute.")

    # -------------------------
    # 3. Pairwise comparisons
    # -------------------------
    log_handle.write("\nPairwise comparisons (paired t-tests):\n")
    sig_pairs = []
    for b1, b2 in combinations(basis_sets, 2):
        t, pval = ttest_rel(pivot[b1], pivot[b2])
        log_handle.write(f"{b1:12} vs {b2:12} -> p = {pval:.4e}\n")
        if pval < 0.05:
            sig_pairs.append(f"{b1} vs {b2} (p={pval:.4e})")

    if sig_pairs:
        conclusions.append("Significant pairwise differences found: " + ", ".join(sig_pairs))
    else:
        conclusions.append("No significant pairwise differences in pairwise t-tests.")

    # -------------------------
    # 4. Mean difference matrix
    # -------------------------
    log_handle.write("\nMean difference matrix (b1 - b2):\n")
    diff_matrix = pd.DataFrame(index=basis_sets, columns=basis_sets)
    for b1 in basis_sets:
        for b2 in basis_sets:
            diff_matrix.loc[b1, b2] = (pivot[b1] - pivot[b2]).mean()
    log_handle.write(diff_matrix.to_string() + "\n")

    # -------------------------
    # 5. Correlation matrix
    # -------------------------
    log_handle.write("\nCorrelation matrix:\n")
    log_handle.write(pivot.corr().to_string() + "\n")

    # -------------------------
    # 6. Variability
    # -------------------------
    log_handle.write("\nStandard deviation per basis:\n")
    log_handle.write(pivot.std().to_string() + "\n")

    # -------------------------
    # 7. Overall significance determination
    # -------------------------
    # Rrequire meaningful effect size or multiple pairwise differences
    eta_sq_threshold = 0.1  # practical significance threshold
    sig_overall = False
    reason = []

    if 'eta_sq' in locals() and aov.anova_table["Pr > F"].iloc[0] < 0.05:
        if eta_sq >= eta_sq_threshold:
            sig_overall = True
            reason.append(f"RM-ANOVA significant with eta^2={eta_sq:.3f}")
        else:
            reason.append(f"RM-ANOVA significant but eta^2={eta_sq:.3f} is small → practically negligible")
    elif p < 0.05:
        if 'eta_sq' in locals() and eta_sq >= eta_sq_threshold:
            sig_overall = True
            reason.append(f"Friedman test significant with eta^2={eta_sq:.3f}")
        else:
            reason.append(f"Friedman test significant but eta^2={eta_sq:.3f} is small → practically negligible")

    if len(sig_pairs) >= 2:
        sig_overall = True
        reason.append(f"{len(sig_pairs)} significant pairwise differences")

    conclusions.append("Overall: " + ("Statistically and practically significant (" + "; ".join(reason) + ")" if sig_overall else "Not practically significant (" + "; ".join(reason) + ")"))

    # -------------------------
    # 8. Log conclusions
    # -------------------------
    log_handle.write("\n" + "-"*60 + "\n")
    log_handle.write("Conclusions:\n")
    for c in conclusions:
        log_handle.write(c + "\n")
    log_handle.write("-"*60 + "\n\n")

    # Store summary
    summary_results[descriptor] = sig_overall

def analyze_per_group(df, descriptor, log_handle):

    log_handle.write("\n" + "#"*60 + "\n")
    log_handle.write(f"PER-GROUP ANALYSIS: {descriptor}\n")
    log_handle.write("#"*60 + "\n")

    # Pivot (same as before, but KEEP NaNs here)
    pivot = df.pivot_table(
        index=["System", "Ring"],
        columns="Basis_Set",
        values=descriptor
    )

    basis_sets = pivot.columns.tolist()

    for (system, ring), row in pivot.iterrows():

        # Skip rows with missing values (optional choice)
        if row.isna().any():
            continue

        log_handle.write("\n" + "-"*50 + "\n")
        log_handle.write(f"Group: {system} | {ring}\n")
        log_handle.write("-"*50 + "\n")

        values = row.values
        bs = basis_sets

        # -------------------------
        # Raw values
        # -------------------------
        for b, v in zip(bs, values):
            log_handle.write(f"{b:12}: {v:.6f}\n")

        # -------------------------
        # Ranking
        # -------------------------
        sorted_pairs = sorted(zip(bs, values), key=lambda x: x[1])
        log_handle.write("\nRanking (low → high):\n")
        for b, v in sorted_pairs:
            log_handle.write(f"{b:12}: {v:.6f}\n")

        # -------------------------
        # Spread
        # -------------------------
        spread = values.max() - values.min()
        log_handle.write(f"\nSpread (max - min): {spread:.6f}\n")

        # -------------------------
        # Mean-centered deviations
        # -------------------------
        mean_val = values.mean()
        log_handle.write("\nDeviation from mean:\n")
        for b, v in zip(bs, values):
            log_handle.write(f"{b:12}: {v - mean_val:+.6f}\n")

        # -------------------------
        # Pairwise differences
        # -------------------------
        log_handle.write("\nPairwise differences (b1 - b2):\n")
        for b1, b2 in combinations(bs, 2):
            diff = row[b1] - row[b2]
            log_handle.write(f"{b1:12} - {b2:12} = {diff:+.6f}\n")

        # -------------------------
        # Conclusion
        # -------------------------
        log_handle.write("\nConclusion:\n")
        log_handle.write(
            f"Basis sets show a spread of {spread:.6f} for this system.\n"
        )

# -----------------------------
# 4. RUN ALL DESCRIPTORS
# -----------------------------
with open(log_file, "w", encoding="utf-8") as f:
    for col in descriptor_cols:
        analyze_descriptor(df, col, f)
        analyze_per_group(df, col, f)

    # -----------------------------
    # 5. FINAL SUMMARY
    # -----------------------------
    f.write("\n" + "="*80 + "\n")
    f.write("FINAL SUMMARY: Significant descriptors\n")
    f.write("="*80 + "\n")
    for desc, significant in summary_results.items():
        if significant:
            f.write(f"{desc}: Statistically significant differences detected.\n")
        else:
            f.write(f"{desc}: No significant differences detected.\n")
    f.write("="*80 + "\n")

print(f"Analysis complete. Results saved to {log_file}")