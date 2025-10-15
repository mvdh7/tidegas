# -*- coding: utf-8 -*-
"""
Reference measurements over universal time axis
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------------------------------------
# 1. Load live logbook
# -------------------------------------------------------------------
csv_path = "DIC_logbook.csv"
df = pd.read_csv(csv_path)

# Ensure numeric
df["Negative removed DIC (umol/L)"] = pd.to_numeric(df["Negative removed DIC (umol/L)"], errors="coerce")
df["Reference"] = df["Reference"].fillna(0).astype(int)

# -------------------------------------------------------------------
# 2. List available dates
# -------------------------------------------------------------------
all_dates = sorted(df["File date"].dropna().unique())
print("Available dates in logbook:")
for d in all_dates:
    print(d)

# -------------------------------------------------------------------
# 3. Prepare universal time axis (hours since midnight)
# -------------------------------------------------------------------
def time_to_hours(time_str):
    """Convert HH:MM:SS to decimal hours."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour + t.minute / 60 + t.second / 3600
    except:
        return None

df_ref = df[df["Reference"] == 1].copy()
df_ref["Hour"] = df_ref["timestamp (UTC)"].apply(time_to_hours)

# -------------------------------------------------------------------
# 4. Plot function
# -------------------------------------------------------------------
def plot_reference_universal(selected_dates=None):
    """
    Plot reference measurements on a universal time axis (hours).
    selected_dates: list of dates to include, default all
    """
    if selected_dates is not None:
        data = df_ref[df_ref["File date"].isin(selected_dates)]
    else:
        data = df_ref
    
    if data.empty:
        print("No reference measurements found.")
        return
    
    plt.figure(figsize=(10, 6))
    
    for date in sorted(data["File date"].unique()):
        day_data = data[data["File date"] == date]
        plt.plot(day_data["Hour"], day_data["Negative removed DIC (umol/L)"],marker='o',label=date)
    
    plt.xlabel("Time (hours)")
    plt.ylabel("Negative removed DIC (umol/L)")
    plt.title("Reference measurements across multiple days")
    plt.xticks(range(8, 16))  # universal 8:00–15:00
    plt.grid(True)
    plt.legend(title="Date")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 5. Example usage
# -------------------------------------------------------------------
selected_dates = ["2025-10-09", "2025-10-10"]  # or None for all
plot_reference_universal(all_dates)
