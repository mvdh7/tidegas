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
import matplotlib.dates as mdates
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
mixing_col = "Mixing and waiting time (seconds)"
acid_totals_to_plot = [0.6,1.2,1.65,2.1,3, 4.2]       # <--- list of total acid volumes to process
#acid_totals_to_plot = np.linspace(0,4.2,15).round(4)
selected_dates = None                  # None or list of specific dates to include
do_regression_lines = True


# -------- Load & preprocess --------
df = pd.read_excel(main_file)
print(f"Loaded {len(df)} rows from {main_file}")
# Sort by date then bottle
# df['date_sort'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
# df.sort_values(by=['date_sort'], inplace=True)
# df.reset_index(drop=True, inplace=True)
# df.drop(columns=['date_sort'], inplace=True)
# Ensure numeric columns
for col in [dic_col, ref_dic_col, waiting_col, acid_col, acid_total_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

# Compute relative DIC (%)
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

#sort out longer mixing time measurements

df = df[df["Mixing and waiting time (seconds)"]==4]
# Exponential decay model
def exp_decay(t, t0, C, k):
    return (100-C) * np.exp(-k * (t-t0)) + C

fit_results = []
# -------- Function to plot for one acid total --------
def plot_acid_total(df, acid_total, use_colorbar_mode=False):
    global fit_results
    df_sub = df[df[acid_total_col] == acid_total].dropna(
        subset=[waiting_col, "Relative DIC (%)", acid_col, date_col]
    )
    if df_sub.empty:
        print(f"⚠️ No data found for acid total = {acid_total} mL.")
        return

    # Convert date to datetime
    df_sub["date_dt"] = pd.to_datetime(df_sub[date_col], format="%d/%m/%Y", errors="coerce")

    # Keep groups with ≥3 points
    group_counts = df_sub.groupby([date_col, acid_col]).size()
    valid_groups = group_counts[group_counts >= 1].index
    df_sub = df_sub.set_index([date_col, acid_col]).loc[valid_groups].reset_index()

    if selected_dates is not None:
        df_sub = df_sub[df_sub[date_col].isin(selected_dates)]

    if df_sub.empty:
        print(f"⚠️ No valid (date, increment) groups for {acid_total} mL.")
        return

    # Sort groups chronologically
    grouped = list(df_sub.groupby([date_col, acid_col]))
    grouped.sort(key=lambda g: pd.to_datetime(g[0][0], format="%d/%m/%Y"))

    fig, ax = plt.subplots(figsize=(10, 6))

    # For colorbar mode
    if use_colorbar_mode:
        cmap = plt.cm.plasma
        norm = plt.Normalize( mdates.date2num(df_sub["date_dt"].min()), mdates.date2num(df_sub["date_dt"].max()))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    # Store fit parameters
    param_lines = []

    colors = plt.cm.tab10.colors
    for i, ((day, inc), group) in enumerate(grouped):
        group = group.sort_values(waiting_col)
        x = group[waiting_col].values
        y = group["Relative DIC (%)"].values
        date_ts = mdates.date2num(group["date_dt"].iloc[0])


        if use_colorbar_mode:
            color = cmap(norm(date_ts))
            marker = "o" if abs(inc - 0.15) < 1e-6 else "x"
            label = None   # legend removed in colorbar mode
        else:
            color = colors[i % len(colors)]
            marker = "o"
            label = f"{day} — {inc:.2f} mL"

        ax.scatter(x, y, color=color, marker=marker, s=70, label=label)
        #ax.errorbar(x,y)
        #ax.plot(x, y, color=color, alpha=0.6)

        # Fit
        if do_regression_lines and len(x) >= 5 and max(group[waiting_col])>=25:
            try:
                p0 = [0, 95, 0.001]
                bounds = ([min(x)-10, 0, 0], [max(x)+10, 120, 1])
                popt, _ = curve_fit(exp_decay, x, y, p0=p0, bounds=bounds, maxfev=10000)

                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, exp_decay(xs, *popt), linestyle='--', color=color, alpha=0.7)

                # Collect fit parameters for side box
                t0, C, k = popt
                param_lines.append(f"{day}, inc={inc:.2f}:  k={k:.4f}, C={C:.1f}")
                t0, C, k = popt
                print(day)
                # ===== STORE C-VALUE =====
                fit_results.append({
                    "date": day,
                    "acid_increment_mL": inc,
                    "acid_total_mL": acid_total,
                    "C_percent": C
                })
                
                print(f"{day}, increment {inc} mL:  C = {C:.2f}%")

          
            except RuntimeError:
                print(f"Fit failed for {day}, inc={inc}")

    # ------------------------- SIDE FIT-PARAMETER BOX -------------------------
    if param_lines:
        textstr = "\n".join(param_lines)
        ax.text(
            0.35, 0.85, textstr,
            transform=ax.transAxes,
            fontsize=14,
            va='center',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )

    # ------------------------- LABELS, COLORBAR, LEGEND -------------------------
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Relative DIC (% of reference)")
    ax.set_title(f"DIC decay — {acid_total:.2f} mL acid total")

    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    ax.grid(True)

    if use_colorbar_mode:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Date")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    else:
        ax.legend(title="Day — Acid increment", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    plt.show()
    

#%%
# -------- Run for all acid totals --------
for acid_total in acid_totals_to_plot:
    plot_acid_total(df, acid_total, use_colorbar_mode=True)


fit_df = pd.DataFrame(fit_results)