# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:56:06 2025

@author: nicor
"""

# -*- coding: utf-8 -*-
"""
Reference measurements over universal time axis
- Color = day (chronological order)
- Marker = Sample type
- Scatter-only, no connecting lines
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
# -------------------------------------------------------------------
# 1. Load live logbook
# -------------------------------------------------------------------
csv_path = "DIC_logbook.csv"
df = pd.read_csv(csv_path)

# -------------------------------------------------------------------
# 1b. Parse File date as real datetimes (day-first) and keep a display label
# -------------------------------------------------------------------
# If File date column already exists as strings like "03/11/2025" (day/month/year)
df["File date dt"] = pd.to_datetime(df["File date"], dayfirst=True, errors="coerce")
# Keep a human friendly label (same format as input)
df["File date label"] = df["File date dt"].dt.strftime("%d/%m/%Y")

# -------------------------------------------------------------------
# 2. Ensure numeric and Reference is integer
# -------------------------------------------------------------------
df["Negative removed DIC (umol/L)"] = pd.to_numeric(df["Negative removed DIC (umol/L)"], errors="coerce")
df["Daily reference DIC (umol/L)"] = pd.to_numeric(df["Daily reference DIC (umol/L)"], errors = "coerce")


df["DIC w.r.t reference (%)"] =100* df["Negative removed DIC (umol/L)"]/df["Daily reference DIC (umol/L)"]
df["Reference"] = df["Reference"].fillna(0).astype(int)


# -------------------------------------------------------------------
# 3. Print available dates (chronologically)
# -------------------------------------------------------------------
all_dates_dt = sorted(df["File date dt"].dropna().unique())
print("Available dates in logbook (chronological):")
for d in all_dates_dt:
    print(d.strftime("%d/%m/%Y"))

# -------------------------------------------------------------------
# 4. Universal time axis helper
# -------------------------------------------------------------------
def time_to_hours(time_str):
    """Convert HH:MM:SS to decimal hours."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour + t.minute / 60 + t.second / 3600
    except Exception:
        return None

# -------------------------------------------------------------------
# 5. Prepare df_ref
# -------------------------------------------------------------------
df_ref = df[df["Reference"] == 1].copy()
df_ref["Hour"] = df_ref["timestamp (UTC)"].apply(time_to_hours)

# Exclusions (keep as before)
#df_ref = df_ref[df_ref["File date label"] != "09/10/2025"]
df_ref = df_ref[df_ref["File date label"] != "18/09/2025"]
df_ref = df_ref[df_ref["Sample type"] != "Crm"]
df_ref = df_ref[df_ref["Sample type"] != "Nuts"]

print(f'mean of DIC w.r.t reference = {np.mean(df_ref["DIC w.r.t reference (%)"])}')
# -------------------------------------------------------------------
# 6. Plot function with corrected chronological ordering for colors
# -------------------------------------------------------------------
def plot_reference_universal(selected_dates=None):
    """
    Scatter-only plot of reference measurements on a universal time axis.
    Colors distinguish days (chronological), markers distinguish sample types.
    selected_dates: list of date strings in 'dd/mm/YYYY' format, or None for all
    """
    if selected_dates is not None:
        # convert selected_dates strings to datetimes for matching
        sel_dt = [pd.to_datetime(d, dayfirst=True) for d in selected_dates]
        data = df_ref[df_ref["File date dt"].isin(sel_dt)].copy()
    else:
        data = df_ref.copy()
    
    if data.empty:
        print("No reference measurements found.")
        return
    
    plt.figure(figsize=(11, 6))
    
    # Unique sample types -> markers (shapes vary by sample type)
    sample_types = sorted(data["Sample type"].dropna().unique())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]  # extend if needed
    marker_map = {stype: markers[i % len(markers)] for i, stype in enumerate(sample_types)}
    
    # Unique days -> chronological colors
    # Get the unique days present in the filtered data, sorted by datetime
    # Unique days -> chronological order
    days_dt_sorted = pd.Series(data["File date dt"].dropna().unique()).sort_values()
    days_labels = days_dt_sorted.dt.strftime("%d/%m/%Y").tolist()
    # Create display labels in the same order
    #days_labels = [d.strftime("%d/%m/%Y") for d in days_dt_sorted]
   
    # Use a vivid full-range colormap and evenly sample across it
    cmap = plt.cm.turbo  # or 'viridis', 'nipy_spectral' if you prefer
    if len(days_dt_sorted) == 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (len(days_dt_sorted) - 1)) for i in range(len(days_dt_sorted))]
    color_map = {days_labels[i]: colors[i] for i in range(len(days_labels))}
    
    # Plot: iterate days in chronological order, then sample types
    for day_dt, day_label in zip(days_dt_sorted, days_labels):
        #print(day_dt)
        day_data = data[data["File date dt"] == day_dt]
        if len(day_data) < 4:
            # keep existing rule: skip days with fewer than 4 points
            continue
        
        for stype in sample_types:
            sdata = day_data[day_data["Sample type"] == stype]
            if sdata.empty:
                continue
            plt.scatter(
                sdata["Hour"],
                sdata["DIC w.r.t reference (%)"],
                marker=marker_map[stype],
                color=color_map[day_label],
                s=70,
                edgecolor="black",
                label=f"{day_label} — {stype}"
            )
    
    plt.xlabel("Time (hours, UTC)")
    plt.ylabel("DIC w.r.t reference (%)")
    plt.title("Reference measurements")
    plt.xticks(range(8, 16))  # universal 8:00–15:00
    plt.grid(True, alpha=0.3)
    plt.legend(title="Date — Sample type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 7. Example usage
# -------------------------------------------------------------------
selected_dates = ["03/11/2025", "05/11/2025", "11/11/2025", "12/11/2025"]  # can be None

plot_reference_universal(selected_dates)
plot_reference_universal(None)  # all chronological dates present in df_ref


# df_ref should already exist (Reference==1, proper Hour column)
df_fit = df_ref.dropna(subset=["Hour", "Negative removed DIC (umol/L)"]).copy()
# X = Hour (0–24), y = Negative removed DIC
X = df_fit["Hour"].values.reshape(-1, 1)
y = df_fit["DIC w.r.t reference (%)"].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Slope and intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"Trend over the day: slope = {slope:.3f} % per hour, intercept = {intercept:.2f} %")

plt.figure(figsize=(10,6))
plt.scatter(df_fit["Hour"], y, alpha=0.5, edgecolor="black", label="Reference DIC")
plt.plot(df_fit["Hour"], y_pred, color="red", linewidth=2, label=f"Linear trend: {slope:.3f} %/h")
plt.xlabel("Time (hours UTC)")
plt.ylabel("DIC w.r.t reference (%)")
plt.title("Reference DIC trend over the day (all samples aggregated)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()