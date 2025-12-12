# -*- coding: utf-8 -*-
"""
Plot Relative DIC (%) vs waiting time (minutes)
Grouped by date & acid increment (mL) with >=3 measurements
Exponential decay fit instead of linear
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 16,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 15,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })


# -------- CONFIG --------
main_file = "Logbook_automated_by_python_testing.xlsx"
date_col = "date"
waiting_col = "waiting time (minutes)"
dic_col = "Calculated DIC (umol/kg)"
ref_dic_col = "Reference DIC (umol/kg)"
acid_col = "acid increments (mL)"   # column for acid volume
selected_dates = None  # None or list of dates to plot
do_regression_lines = True

# -------- Load --------
df = pd.read_excel(main_file)
print(f"Loaded {len(df)} rows from {main_file}")

# Ensure numeric columns
for col in [dic_col, ref_dic_col, waiting_col, acid_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

# -------- Compute relative DIC (%) --------
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

# -------- Filter for valid groups --------
plot_df = df.dropna(subset=[waiting_col, "Relative DIC (%)", acid_col, date_col])
group_counts = plot_df.groupby([date_col, acid_col]).size()
valid_groups = group_counts[group_counts >= 3].index
plot_df = plot_df.set_index([date_col, acid_col]).loc[valid_groups].reset_index()

if plot_df.empty:
    raise RuntimeError("No valid groups (>=3 points per day per acid increment).")
print(f"📈 Found {len(valid_groups)} valid (date, increment) sets for plotting.")

# -------- Select dates --------
all_dates = sorted(plot_df[date_col].unique())
if selected_dates is None:
    selected_dates = all_dates
plot_df = plot_df[plot_df[date_col].isin(selected_dates)]

# -------- Exponential decay function --------
def exp_decay(t, R0, k):
    return R0 * np.exp(-k * t)

# -------- Plotting --------
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors

for i, ((day, inc), group) in enumerate(plot_df.groupby([date_col, acid_col])):
    group = group.sort_values(waiting_col)
    x = group[waiting_col].values
    y = group["Relative DIC (%)"].values
    color = colors[i % len(colors)]
    label = f"{day} — {inc:.2f} mL"

    plt.scatter(x, y, label=label, color=color)
    plt.plot(x, y, color=color, alpha=0.6)

    # Exponential fit
    if do_regression_lines and len(x) >= 3:
        try:
            popt, _ = curve_fit(exp_decay, x, y, p0=(100, 0.001), maxfev=10000)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, exp_decay(xs, *popt), linestyle='--', color=color, alpha=0.7)
            plt.text(xs.mean()+25, exp_decay(xs.mean(), *popt)+15,
                     f"k={popt[1]:.4f} 1/min", color=color, fontsize=12,
                     ha='center', va='top', backgroundcolor='white')
        except RuntimeError:
            print(f"⚠️ Could not fit exponential for {label}")
    print(y[0], popt)
plt.xlabel("Waiting time (minutes)")
plt.ylabel("Relative DIC (% of reference)")
plt.title("DIC decay vs waiting time — 4.2 mL acid total")
plt.axhline(100, color='gray', linestyle='--', linewidth=1)
plt.legend(title="Day — Acid increment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
