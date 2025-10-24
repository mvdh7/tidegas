import numpy as np
import pandas as pd
import os
import re

# -----------------------------
# Folder path and file selection
# -----------------------------
folder_path = "data/vindta/r2co2/Bobometer"

# Get all txt files
all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# Sort files by the number inside parentheses, e.g., "co2data (3).txt"
def extract_number(filename):
    match = re.search(r"\((\d+)\)", filename)
    return int(match.group(1)) if match else -1

all_files = sorted(all_files, key=extract_number)

# Optionally, select a subset of files (e.g., only files 1-5)
selected_files = all_files[0:92]  # Change indices as needed
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

# -----------------------------
# Example plotting function
# -----------------------------
import matplotlib.pyplot as plt

def plot_batch(df, column_name, title, ylabel, xlabel="Index"):
    plt.figure(figsize=(10,6))
    for idx, row in df.iterrows():
        plt.plot(row[column_name], label=row["file_name"],linewidth =2 )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    #plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
plot_batch(df_results, "total_cell_integrated", 
           "Integrated Cell Current (Raw)", "Integrated current [µA·s]", "Time [s]")
plot_batch(df_results, "total_cell_negative_removed", 
           "Integrated Cell Current (Negative Removed)", "Integrated current [µA·s]", "Time [s]")
plot_batch(df_results, "negative_removed_difference", 
           "Change in Integrated Current After Negative Removal", "Δ Current [µA·s]")

plot_batch(df_results, 
           column_name="slope", 
           title="slope - Batch Comparison", 
           ylabel="Δ Current [µA·s]")