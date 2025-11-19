# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:17:05 2025

@author: nicor
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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
# Column names (exact strings from your dataset)

# Column names
date_col = "date"
sample_type_col = "sample/junk"
ref_flag_col = "Alkalinity daily reference measurement"
calc_alk_col = "calkulate alkalinity"

# New output columns
mean_col = "alkalinity daily reference alkalinity (umol/kg)"
std_col  = "alkalinity daily reference std (umol/kg)"

# Convert dates
df[date_col] = pd.to_datetime(df[date_col],dayfirst = True)

# Identify only reference rows
ref_mask = df[ref_flag_col] == 1

# Group by date AND sample type
group_keys = [df[date_col].dt.date, df[sample_type_col]]

# Compute daily stats only for flagged rows
daily_mean = df.loc[ref_mask].groupby(group_keys)[calc_alk_col].transform("mean")
daily_std  = df.loc[ref_mask].groupby(group_keys)[calc_alk_col].transform("std")

# Write results into the new columns but only for flagged rows
df.loc[ref_mask, mean_col] = daily_mean.round(3)
df.loc[ref_mask, std_col]  = daily_std.round(3)

# Save
df.to_csv("updated_file.csv", index=False)

print("Daily reference alkalinity mean/std calculated per date & sample type.")

