# -*- coding: utf-8 -*-
"""
Plot Relative DIC (%) vs waiting time (minutes)
Grouped by acid increment (mL) with >=3 measurements
Exponential decay fit with k values displayed in color near the curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------- CONFIG --------
main_file = "Logbook_automated_by_python_testing.xlsx"
waiting_col = "waiting time (minutes)"
dic_col = "Calculated DIC (umol/kg)"
ref_dic_col = "Reference DIC (umol/kg)"
acid_col = "acid increments (mL)"  
do_regression_lines = True
x_offset = 1.0  # minutes offset for k label
y_offset = 1.0  # % offset for k label

# -------- Load --------
df = pd.read_excel(main_file)
for col in [dic_col, ref_dic_col, waiting_col, acid_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -------- Relative DIC (%) --------
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

# -------- Filter valid groups (>=3 points) --------
plot_df = df.dropna(subset=[waiting_col, "Relative DIC (%)", acid_col])
group_counts = plot_df.groupby([acid_col]).size()
valid_groups = group_counts[group_counts >= 3].index
plot_df = plot_df[plot_df[acid_col].isin(valid_groups)]

if plot_df.empty:
    raise RuntimeError("No valid groups (>=3 points per acid increment).")
print(f"📈 Found {len(valid_groups)} valid acid increment sets for plotting.")

# -------- Exponential decay function --------
def exp_decay(t, R0, k):
    return R0 * np.exp(-k * t)

# -------- Plotting --------
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20.colors

for i, (inc, group) in enumerate(plot_df.groupby(acid_col)):
    group = group.sort_values(waiting_col)
    x = group[waiting_col].values
    y = group["Relative DIC (%)"].values
    color = colors[i % len(colors)]
    label = f"{inc:.1f} mL"

    # Plot points and lines
    plt.scatter(x, y, label=label, color=color)
    plt.plot(x, y, color=color, alpha=0.6)

    # Exponential fit
    if do_regression_lines and len(x) >= 3:
        try:
            popt, pcov = curve_fit(exp_decay, x, y, p0=(100, 0.001), maxfev=10000)
            xs = np.linspace(x.min(), x.max(), 100)
            plt.plot(xs, exp_decay(xs, *popt), linestyle='--', color=color, alpha=0.7)

            # extract k and its standard error
            R0, k = popt
            perr = np.sqrt(np.diag(pcov))
            k_err = perr[1]

            # Position the k label slightly right and above the last data point
            plt.text(x[-1]+x_offset, y[-1]+y_offset, f"k={k:.4f}±{k_err:.4f}", 
                     color=color, fontsize=9, ha='left', va='bottom', backgroundcolor='white', alpha=0.8)

        except RuntimeError:
            print(f"⚠️ Could not fit exponential for increment {label}")

plt.xlabel("Waiting time (minutes)")
plt.ylabel("Relative DIC (% of reference)")
plt.title("DIC decay vs waiting time — grouped by acid increment")
plt.axhline(100, color='gray', linestyle='--', linewidth=1)
plt.legend(title="Acid increment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
