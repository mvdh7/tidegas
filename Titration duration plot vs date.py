# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:43:20 2025

@author: nicor
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Setup plotting parameters to make everything bigger and nicer
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20
})

# Read Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure required columns exist
required_cols = ["Titration duration (seconds)", "date"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing values in required columns
plot_df = df.dropna(subset=required_cols).copy()

# Convert 'date' to datetime
plot_df["Date"] = pd.to_datetime(plot_df["date"], dayfirst=True)
#select only the files without any (significant) waiting time 
#plot_df =plot_df[plot_df["waiting time (minutes)"]<=0.05]
plot_df =plot_df[plot_df["acid increments (mL)"]<=0.15]
plot_df =plot_df[plot_df["acid added (mL)"]==4.2]
plot_df = plot_df[plot_df["date"]!="09/10/2025"]
plot_df = plot_df[plot_df["Mixing and waiting time (seconds)"]==4]
plot_df_bottle_1 = plot_df[plot_df["bottle"]==1]
plot_df_other_bottles = plot_df[plot_df["bottle"]!=1]

# Scatter plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(
    plot_df_bottle_1["Date"], 
    plot_df_bottle_1["Titration duration (seconds)"], 
    yerr=10,               # ±10 seconds error
    fmt='x',               # cross markers
    ecolor='red',          # error bar color
    capsize=5,             # small cap at end of error bars
    markersize=6,
    linestyle='none',
    color='red',
    label = "first titration"
)
plt.errorbar(
    plot_df_other_bottles["Date"], 
    plot_df_other_bottles["Titration duration (seconds)"], 
    yerr=10,               # ±10 seconds error
    fmt='o',               # circle markers
    ecolor='blue',          # error bar color
    capsize=5,             # small cap at end of error bars
    markersize=6,
    linestyle='none',
    color='blue',
    label = "other titrations"
)
plt.legend()
# Formatting
plt.xlabel("Date")
plt.ylabel("Titration Duration (seconds)")
plt.title("Titration Duration vs Date with ±10 Seconds Error")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
#plt.gcf().autofmt_xdate(rotation=45)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

plot_df =plot_df[plot_df["bottle"]!=1]
# Scatter plot with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(
    plot_df["Date"], 
    plot_df["Titration duration (seconds)"], 
    yerr=10,               # ±10 seconds error
    fmt='o',               # circle markers
    ecolor='blue',          # error bar color
    capsize=5,             # small cap at end of error bars
    markersize=6,
    linestyle='none',
    color='blue'
)

# Formatting
plt.xlabel("Date")
plt.ylabel("Titration Duration (seconds)")
plt.title("Titration Duration vs Date with ±10 Seconds Error")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
plt.gcf().autofmt_xdate(rotation=45)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()