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
plot_df = plot_df[plot_df["date"]=="11/11/2025"]

plot_df = plot_df[plot_df["bottle"]!="1"]

plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "10-11-2025"  # YYYY-MM-DD
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date


#plot_df = plot_df[plot_df["batch"]==1]

plot_df["Titrant Volume (ml)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Calculated in situ DIC (umol/kg)"] =  pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"], errors='coerce')
plot_df["DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC (%)"] = pd.to_numeric(100*plot_df["Calculated in situ DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC difference (umol)"] = pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"]-plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting Time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration Duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')

#%%
# Scatter plot with hue by date
plt.figure()
plt.grid(True,alpha =0.4)
sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    palette="tab20",
    s=70,
    label = "Measured DIC"
)

plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["DIC (%)"],yerr = 1,fmt="none", capsize=4, alpha=0.7)
sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    palette="tab20",
    s=70,
    label = "In-situ DIC"
)
plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["In-situ DIC (%)"],yerr = 1,fmt="none", capsize=4, alpha=0.7)
plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")
plt.title("DIC vs Acid Added (11/11)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#%%

import seaborn as sns
from scipy.optimize import curve_fit

# --- Exponential model ---
def exp_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Prepare data ---
fit_df = plot_df.sort_values(by="Titrant Volume (ml)")

# Include all points (no trimming)
x = fit_df["Titrant Volume (ml)"].values
y = fit_df["In-situ DIC (%)"].values

# Fit 3rd-order polynomial: y = ax^3 + bx^2 + cx + d
coeffs = np.polyfit(x[:-2], y[:-2], 4)   # returns [a, b, c, d]
poly = np.poly1d(coeffs)       # convenient polynomial object

# Generate new x array from 0 → 4.05
x_new = np.linspace(0, 4.05, 28)
y_fit = poly(x_new)


# --- Convert % DIC to absolute DIC ---
ref_dic = plot_df["Reference DIC (umol/kg)"].iloc[0]
absolute_dic = (y_fit / 100) * ref_dic

# --- Store in dataframe ---
fit_df_out = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": absolute_dic
})

#%%
# Put in dataframe for convenience
fit_df = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": absolute_dic
})

fit_df.head()
plt.figure()
plt.grid(True, alpha=0.4)

sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    s=70,
    label="Measured DIC"
)

plt.errorbar(
    x=plot_df["Titrant Volume (ml)"],
    y=plot_df["DIC (%)"],
    yerr=1,
    fmt="none",
    capsize=4,
    alpha=0.7
)

sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    s=70,
    label="In-situ DIC"
)

plt.errorbar(
    x=plot_df["Titrant Volume (ml)"],
    y=plot_df["In-situ DIC (%)"],
    yerr=1,
    fmt="none",
    color= "orange",
    capsize=4,
    alpha=0.7
)

# Plot fitted exponential model
plt.plot(
    fit_df_out["Titrant Volume (ml)"],
    fit_df_out["Fitted In-situ DIC (%)"],
    label="Exponential fit",
    linewidth=2
)

plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")
plt.title("DIC vs Acid Added (11/11)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
