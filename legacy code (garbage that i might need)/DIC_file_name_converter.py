import os
import pandas as pd
import re
from datetime import datetime

FARADAY = 96485.3321  # C/mol

# Paths
folder_path = "data/vindta/r2co2/Bobometer"
main_csv = "DIC_logbook.csv"
out_csv = "main_table_updated.csv"

# -------------------------------------------------------------------
# 1. Rename new-format files (e.g., co2data - 2025-10-09T140222.854.txt)
# -------------------------------------------------------------------

def rename_new_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    # Match files of form: co2data - 2025-10-09T140222.854.txt
    timestamp_pattern = re.compile(r"co2data\s*-\s*(\d{4}-\d{2}-\d{2}T\d{6}\.\d{3})\.txt")

    new_files = []
    for f in files:
        match = timestamp_pattern.match(f)
        if match:
            ts = match.group(1)
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H%M%S.%f")
                new_files.append((f, dt))
            except ValueError:
                continue

    # Sort by timestamp
    new_files.sort(key=lambda x: x[1])

    # Group files by date
    grouped = {}
    for original_name, dt in new_files:
        date_str = dt.strftime("%d-%m")
        grouped.setdefault(date_str, []).append((original_name, dt))

    # Rename with date-based pattern
    for date_str, files_for_day in grouped.items():
        for i, (orig_name, dt) in enumerate(files_for_day, start=1):
            new_name = f"co2data {date_str} ({i}).txt"
            src = os.path.join(folder, orig_name)
            dst = os.path.join(folder, new_name)

            if not os.path.exists(dst):
                os.rename(src, dst)
                print(f"Renamed: {orig_name} → {new_name}")
            else:
                print(f"Skipped (already exists): {dst}")

rename_new_files(folder_path)
