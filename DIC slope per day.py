from scipy.stats import linregress
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
folder_path = "data/vindta/r2co2/Bobometer"
SAMPLE_RATE = 1   # seconds per sample
FIT_WINDOW = 180  # seconds (last N seconds)

# -----------------------------
# Collect all .txt files
# -----------------------------
all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

def extract_date_and_number(filename):
    """Extract date (as datetime) and index number from file name."""
    date_match = re.search(r"(\d{1,2}-\d{1,2})", filename)
    if date_match:
        date_str = date_match.group(1)
        date_obj = datetime.strptime(date_str + "-2000", "%d-%m-%Y")  # arbitrary year
    else:
        date_obj = datetime(2000, 1, 1)

    number_match = re.search(r"\((\d+)\)\.txt$", filename)
    number = int(number_match.group(1)) if number_match else -1
    return (date_obj, number)

# Sort files chronologically, then by index
all_files = sorted(all_files, key=extract_date_and_number)
selected_files = all_files[101:]  # optionally slice if needed

# -----------------------------
# Process each file
# -----------------------------
results = {
    "file_name": [],
    "date": [],
    "fit_slope": [],
    "fit_intercept": [],
    "fit_r2": [],
    "final_difference": [],
}

for file_name in selected_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        file = pd.read_csv(file_path)
    except Exception as e:
        print(f"Skipping {file_name}: {e}")
        continue

    # Spike correction
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
        if total_cell_integrated[i + 1] <= total_cell_integrated[i]:
            pass
        else:
            integrated_current += total_cell_integrated[i + 1] - total_cell_integrated[i]
        total_cell_negative_removed.append(integrated_current)
    total_cell_negative_removed = np.array(total_cell_negative_removed)

    # Compute the difference between final integrated and cleaned currents
    final_difference = float(total_cell_negative_removed[-1] - total_cell_integrated[-1])

    # Linear regression on last FIT_WINDOW seconds of cleaned current
    y = total_cell_negative_removed
    n_points = len(y)
    t = np.arange(n_points) * SAMPLE_RATE

    if n_points > FIT_WINDOW:
        t_fit = t[-FIT_WINDOW:]
        y_fit = y[-FIT_WINDOW:]
    else:
        t_fit = t
        y_fit = y

    slope, intercept, r_value, p_value, std_err = linregress(t_fit, y_fit)

    # Extract date string (e.g. "03-11")
    date_match = re.search(r"(\d{1,2}-\d{1,2})", file_name)
    date_str = date_match.group(1) if date_match else "01-01"

    # Store results
    results["file_name"].append(file_name)
    results["date"].append(date_str)
    results["fit_slope"].append(slope)
    results["fit_intercept"].append(intercept)
    results["fit_r2"].append(r_value ** 2)
    results["final_difference"].append(final_difference)

# -----------------------------
# Convert to DataFrame
# -----------------------------
df_results = pd.DataFrame(results)

# -----------------------------
# Aggregate by day with mean and std
# -----------------------------
df_daily = (
    df_results.groupby("date")
    .agg(
        avg_slope=("fit_slope", "mean"),
        std_slope=("fit_slope", "std"),
        avg_final_difference=("final_difference", "mean"),
        std_final_difference=("final_difference", "std"),
        n_files=("file_name", "count"),
    )
    .reset_index()
    .sort_values("date", key=lambda x: x.apply(lambda d: datetime.strptime(d, "%d-%m")))
)

# -----------------------------
# Plotting
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# === 1️⃣ Violin Plot for slope distribution ===
plt.figure(figsize=(10,6))
dates_sorted = df_results["date"].unique()
sorted_dates = sorted(dates_sorted, key=lambda d: datetime.strptime(d, "%d-%m"))

data = [df_results.loc[df_results["date"] == d, "fit_slope"].values for d in sorted_dates]
plt.violinplot(data, showmeans=True)
plt.xticks(range(1, len(sorted_dates)+1), sorted_dates, rotation=45)
plt.title(f"Slope Distribution per Day (Last {FIT_WINDOW}s)")
plt.ylabel("Slope [µA·s per s]")
plt.tight_layout()
plt.show()

# === 2️⃣ Line Plot with error bands (Slope) ===
plt.figure(figsize=(10,6))
plt.plot(df_daily["date"], df_daily["avg_slope"], "o-", label="Average slope", color="C0")
plt.fill_between(
    df_daily["date"],
    df_daily["avg_slope"] - df_daily["std_slope"],
    df_daily["avg_slope"] + df_daily["std_slope"],
    color="C0",
    alpha=0.2,
    label="±1 SD"
)
plt.xticks(rotation=45)
plt.title(f"Average Slope per Day (±1 SD, Last {FIT_WINDOW}s)")
plt.xlabel("Date")
plt.ylabel("Slope [µA·s per s]")
plt.legend()
plt.tight_layout()
plt.show()

# === 3️⃣ Line Plot with error bands (Final difference) ===
plt.figure(figsize=(10,6))
plt.plot(df_daily["date"], df_daily["avg_final_difference"], "s-", label="Average final difference", color="orange")
plt.fill_between(
    df_daily["date"],
    df_daily["avg_final_difference"] - df_daily["std_final_difference"],
    df_daily["avg_final_difference"] + df_daily["std_final_difference"],
    color="orange",
    alpha=0.25,
    label="±1 SD"
)
plt.xticks(rotation=45)
plt.title("Average Final Difference per Day (Negative Removed − Integrated)")
plt.xlabel("Date")
plt.ylabel("Δ Current [µA·s]")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: print summary table
print(df_daily)
