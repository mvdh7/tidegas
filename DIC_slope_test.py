from scipy.stats import linregress
import numpy as np
import pandas as pd
import os
import re

import matplotlib.pyplot as plt

folder_path = "data/vindta/r2co2/Bobometer"

# Get all txt files
all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# Sort files by the number inside parentheses, e.g., "co2data (3).txt"
def extract_number(filename):
    match = re.search(r"\((\d+)\)", filename)
    return int(match.group(1)) if match else -1

all_files = sorted(all_files, key=extract_number)

# Optionally, select a subset of files (e.g., only files 1-5)
selected_files = all_files# [0:20]  # Change indices as needed
# selected_files.remove("co2data (60).txt")
# selected_files.append("co2data (60).txt")
# -----------------------------
# Initialize results dictionary
# -----------------------------
results = {
    "file_name": [],
    "total_cell_integrated": [],
    "total_cell_negative_removed": [],
    "negative_removed_difference": [],
    "slope": [],
}

# -----------------------------
# Batch processing loop
# -----------------------------
for file_name in selected_files:
    file_path = os.path.join(folder_path, file_name)
    file = pd.read_csv(file_path)

    # Spike correction (mask peaks above 1e7)
    file[" Cell[uAs]"] = file[" Cell[uAs]"].mask(
        file[" Cell[uAs]"] >= 1e7,
        (file[" Cell[uAs]"].shift(1) + file[" Cell[uAs]"].shift(-1)) / 2
    )

    # Total integrated current
    total_cell_integrated = file[" Cell[uAs]"].tolist()

    # Remove negative currents
    total_cell_negative_removed = [0]
    integrated_current = 0
    for i in range(len(total_cell_integrated) - 1):
        if file[" Cell[uAs]"][i + 1] <= file[" Cell[uAs]"][i]:
            integrated_current = integrated_current
            
        else:
            integrated_current += file[" Cell[uAs]"][i + 1] - file[" Cell[uAs]"][i]
        total_cell_negative_removed.append(integrated_current)
    total_cell_negative_removed = np.array(total_cell_negative_removed)

    # Difference between cleaned and raw
    negative_removed_difference = total_cell_negative_removed - np.array(total_cell_integrated)

    # Slope of positive changes
    slope = pd.Series(negative_removed_difference).diff()
    slope = slope[slope > 0.0001].values

    # Store results
    results["file_name"].append(file_name)
    results["total_cell_integrated"].append(total_cell_integrated)
    results["total_cell_negative_removed"].append(total_cell_negative_removed)
    results["negative_removed_difference"].append(negative_removed_difference)
    results["slope"].append(slope)

# -----------------------------
# Convert results to DataFrame
# -----------------------------
df_results = pd.DataFrame(results)

# Sampling rate assumption (1 sample per second, adjust if different)
SAMPLE_RATE = 1  # seconds per sample
FIT_WINDOW = 180  # seconds (last N seconds)

# Add new columns to results DataFrame
df_results["fit_slope"] = np.nan
df_results["fit_intercept"] = np.nan
df_results["fit_r2"] = np.nan

# Perform the linear fits for each file
for idx, row in df_results.iterrows():
    y = np.array(row["total_cell_negative_removed"])
    n_points = len(y)
    
    # Create time array (in seconds)
    t = np.arange(n_points) * SAMPLE_RATE
    
    # Select last 180 seconds (or fewer if the file is shorter)
    if n_points > FIT_WINDOW:
        t_fit = t[-FIT_WINDOW:]
        y_fit = y[-FIT_WINDOW:]
    else:
        t_fit = t
        y_fit = y
    
    # Linear regression on that segment
    slope, intercept, r_value, p_value, std_err = linregress(t_fit, y_fit)
    
    # Save results
    df_results.loc[idx, "fit_slope"] = slope
    df_results.loc[idx, "fit_intercept"] = intercept
    df_results.loc[idx, "fit_r2"] = r_value**2

# ==========================
# PLOT: Linear fit overlays
# ==========================

def plot_fits(df, n_files=5):
    """Plot the last 180s fits for a subset of files."""
    plt.figure(figsize=(10,6))
    
    for idx, row in df.head(n_files).iterrows():
        y = np.array(row["total_cell_negative_removed"])
        n_points = len(y)
        t = np.arange(n_points) * SAMPLE_RATE
        
        # Fit segment
        if n_points > FIT_WINDOW:
            t_fit = t[-FIT_WINDOW:]
            y_fit = y[-FIT_WINDOW:]
        else:
            t_fit = t
            y_fit = y
        
        # Linear fit line
        y_line = row["fit_slope"] * t_fit + row["fit_intercept"]
        
        plt.plot(t, y, alpha=0.5, label=f"{row['file_name']} (data)")
        plt.plot(t_fit, y_line, '--', linewidth=2, label=f"{row['file_name']} fit")
    
    plt.title(f"Linear Fit of Last {FIT_WINDOW} Seconds (Negative Removed)")
    plt.xlabel("Time [s]")
    plt.ylabel("Integrated current [µA·s]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_fits(df_results, n_files=5)


# Helper to extract file number from name
def extract_number(filename):
    match = re.search(r"\((\d+)\)", filename)
    return int(match.group(1)) if match else np.nan

# Add a "file_number" column if not present
if "file_number" not in df_results.columns:
    df_results["file_number"] = df_results["file_name"].apply(extract_number)

# Sort by file number (optional, for cleaner x-axis)
df_results = df_results.sort_values("file_number")

# --- PLOT ---
plt.figure(figsize=(10,6))
plt.plot(df_results["file_number"], df_results["fit_slope"], "o-", linewidth=2)
plt.title("Linear Fit Slope (Last 180 s) Across Files")
plt.xlabel("File Number")
plt.ylabel("Slope of Last 180 s [µA·s per s]")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
