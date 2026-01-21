# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:54:40 2025

@author: nicor
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:31:57 2025

@author: nicor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline

#setup plotting parameters to make everything bigger
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 18,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 18,    # x tick labels
    "ytick.labelsize": 18,    # y tick labels
    "legend.fontsize": 16,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure columns exist
if not all(col in df.columns for col in ["Calculated DIC (umol/kg)", "acid added (mL)", "date"]):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = df.dropna(subset=["Calculated DIC (umol/kg)", "acid added (mL)", "date", "waiting time (minutes)"]).copy()

#select only the files without any (significant) waiting time 
plot_df =plot_df[plot_df["waiting time (minutes)"]<=0.05]
plot_df =plot_df[plot_df["acid increments (mL)"]<=0.15]
plot_df =plot_df[plot_df["Mixing and waiting time (seconds)"]==4]

plot_df = plot_df[plot_df["bottle"]!="1"]

plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "10-22-2025"  # MM-DD-YYYY
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date. Excluded dates where bobometer was misbehaving due to nitrogen leaks 
plot_df = plot_df[plot_df["date"]!= "10/11/2025"]
plot_df = plot_df[plot_df["date"]!= "7/11/2025"]
plot_df = plot_df[plot_df["date"]!= "5/11/2025"]



plot_df["Titrant Volume (ml)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Calculated in situ DIC (umol/kg)"] =  pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"], errors='coerce')
plot_df["DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC (%)"] = pd.to_numeric(100*plot_df["Calculated in situ DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC difference (umol)"] = pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"]-plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting Time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration Duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')

#load errors from seperate file 
dic_unc = pd.read_csv("DIC_total_uncertainty_vs_titrant_volume.csv")
plot_df = plot_df.merge(
    dic_unc,
    left_on="Titrant Volume (ml)",
    right_on="titrant_volume_mL",
    how="left"
)
#%%
# Scatter plot with hue by date
plt.figure()
plt.grid(True,alpha =0.4)
plt.scatter(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    s=50,
    label = "Measured DIC"
)

plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["DIC (%)"],yerr = 0.236,fmt="none", capsize=4, alpha=0.7)
plt.scatter(
    data=plot_df,color = 'red',
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    s=50,
    label = "In-situ DIC"
)
plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["In-situ DIC (%)"],color = 'red', yerr = plot_df["total_DIC_uncertainty_percent"],fmt="none", capsize=4, alpha=0.7)
plt.xlabel("Titrant Volume (mL)")
plt.ylabel("Remaining DIC (%)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#%%

plot_df = plot_df[plot_df["Titrant Volume (ml)"]!=4.8] # remove 4.8 mL acid
# --- Prepare data: group by x ---
grouped = plot_df.groupby("Titrant Volume (ml)")["In-situ DIC (%)"]

x_unique = grouped.mean().index.values
y_mean = grouped.mean().values
y_std = grouped.std().values  # standard deviation

# Optional: if any std is NaN (single measurement), replace with small value
y_std = np.nan_to_num(y_std, nan=1.0)

# --- Fit cubic polynomial to mean values ---
# Use weights = 1 / std to account for variability
coeffs = np.polyfit(x_unique, y_mean, 4, w=1/y_std)
poly = np.poly1d(coeffs)

# --- Evaluate fit on fine x-grid ---
x_new = np.linspace(0, 4.2, 29)
y_fit = poly(x_new)

# --- Convert % DIC to absolute DIC ---
ref_dic = plot_df["Reference DIC (umol/kg)"].iloc[0]
absolute_dic = (y_fit / 100.0) * ref_dic

# --- Store results ---
fit_df_out = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": absolute_dic
})
fit_df_out = fit_df_out[fit_df_out["Titrant Volume (ml)"]!=4.8]
# --- Optional: plot ---
plt.figure()
plt.grid(True, alpha=0.4)

# Scatter all original points
sns.scatterplot(data=plot_df, x="Titrant Volume (ml)", y="In-situ DIC (%)",
                s=70, label="In-situ DIC")

# Error bars for grouped means
plt.errorbar(x_unique, y_mean, yerr=y_std, fmt='o', color='red', capsize=4,
             label="Mean ± SD")

# Polynomial fit
plt.plot(x_new, y_fit, label="fit to mean", linewidth=3, color="Black")

plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")



# setup for writing the equation in the plot
terms = [
    (coeffs[0], "x⁴"),
    (coeffs[1], "x³"),
    (coeffs[2], "x²"),
    (coeffs[3], "x"),
    (coeffs[4], "")
]

eqn = "y = "
for i, (c, term) in enumerate(terms):
    sign = "−" if c < 0 else "+"
    mag = abs(c)

    # First term: no leading "+"
    if i == 0:
        eqn += f"{mag:.2f}{term} "
    else:
        eqn += f"{sign} {mag:.2f}{term} "
# --- Add equation to plot (bottom left) ---
plt.text(
    0.02, 0.05, eqn,
    transform=plt.gca().transAxes,  # axis-relative positioning
    fontsize=14,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
)
plt.legend()
plt.tight_layout()
plt.show()
print(coeffs)
