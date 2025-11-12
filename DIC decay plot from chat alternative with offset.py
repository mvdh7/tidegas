# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:15:35 2025

@author: nicor
"""

# -*- coding: utf-8 -*-
"""
Plot Relative DIC (%) vs waiting time (minutes)
Grouped by date & acid increment (mL) with >=3 measurements
Exponential decay fit instead of linear
Handles multiple acid totals automatically
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------- STYLE --------
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})


# -------- CONFIG --------
main_file = "Logbook_automated_by_python_testing.xlsx"
date_col = "date"
waiting_col = "waiting time (minutes)"
dic_col = "Calculated DIC (umol/kg)"
ref_dic_col = "Reference DIC (umol/kg)"
acid_col = "acid increments (mL)"      # incremental acid volume
acid_total_col = "acid added (mL)"     # total acid added per run

acid_totals_to_plot = [0.6,1.2,1.65,2.1,3, 4.2]       # <--- list of total acid volumes to process
selected_dates = None                  # None or list of specific dates to include
do_regression_lines = True


# -------- Load & preprocess --------
df = pd.read_excel(main_file)
print(f"Loaded {len(df)} rows from {main_file}")

# Ensure numeric columns
for col in [dic_col, ref_dic_col, waiting_col, acid_col, acid_total_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

# Compute relative DIC (%)
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

# Exponential decay model
def exp_decay(t, t0, C, k):
    return (100-C) * np.exp(-k * (t-t0)) + C


# -------- Function to plot for one acid total --------
def plot_for_acid_total(df, acid_total):
    # Filter dataset
    df_sub = df[df[acid_total_col] == acid_total].dropna(
        subset=[waiting_col, "Relative DIC (%)", acid_col, date_col]
    )

    if df_sub.empty:
        print(f"⚠️ No data found for acid total = {acid_total} mL.")
        return

    # Keep only groups with ≥3 points
    group_counts = df_sub.groupby([date_col, acid_col]).size()
    valid_groups = group_counts[group_counts >= 1].index
    df_sub = df_sub.set_index([date_col, acid_col]).loc[valid_groups].reset_index()

    if df_sub.empty:
        print(f"⚠️ No valid (date, increment) groups for {acid_total} mL.")
        return

    # Select dates if specified
    if selected_dates is not None:
        df_sub = df_sub[df_sub[date_col].isin(selected_dates)]

    # -------- Plot --------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, ((day, inc), group) in enumerate(df_sub.groupby([date_col, acid_col])):
        group = group.sort_values(waiting_col)
        x = group[waiting_col].values
        y = group["Relative DIC (%)"].values
        color = colors[i % len(colors)]
        label = f"{day} — {inc:.2f} mL"

        plt.scatter(x, y, label=label, color=color)
        plt.plot(x, y, color=color, alpha=0.6)

        # Exponential fit
        if do_regression_lines and len(x) >= 5:
            try:
                # Initial guesses
                p0 = [0, 95, 0.001]  # [t0, C, k]
                # Bounds: [t0, C, k]
                lower_bounds = [min(x) - 10, 0, 0]
                upper_bounds = [max(x) + 10, 120, 1]
        
                popt, _ = curve_fit(
                    exp_decay, x, y, p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=10000
                )
        
                xs = np.linspace(x.min(), x.max(), 100)
                plt.plot(xs, exp_decay(xs, *popt), linestyle='--', color=color, alpha=0.7)
        
                # Nicely formatted text
                t0, C, k = popt
                plt.text(xs.mean()+35-xs.max()/6, exp_decay(xs.mean(), *popt) + 15,
                         f"k={k:.4f} 1/min,C={C:.1f}%", color=color, fontsize=12,
                         ha='center', va='bottom', backgroundcolor='white')
        
                print(f"{label}: t0={t0:.2f}, C={C:.2f}, k={k:.4f}")
        
            except RuntimeError:
                print(f"⚠️ Could not fit exponential for {label}")

    plt.xlabel("Waiting time (minutes)")
    plt.ylabel("Relative DIC (% of reference)")
    plt.title(f"DIC decay vs waiting time — {acid_total:.1f} mL acid total")
    plt.axhline(100, color='gray', linestyle='--', linewidth=1)
    plt.legend(title="Day — Acid increment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------- Run for all acid totals --------
for acid_total in acid_totals_to_plot:
    print(f"\n=== Plotting for acid total {acid_total} mL ===")
    plot_for_acid_total(df, acid_total)
