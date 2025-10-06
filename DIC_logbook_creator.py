import os
import pandas as pd
import re

FARADAY = 96485.3321  # C/mol

# Paths
folder_path = "data/vindta/r2co2/Bobometer"
main_csv = "DIC_logbook.csv"   # your existing table
out_csv = "main_table_updated.csv"

# Load the main table
main_table = pd.read_csv(main_csv)

# Function to extract number from filename
def extract_number(fname):
    match = re.search(r"\((\d+)\)", fname)  # matches "co2data (15).txt" → 15
    return int(match.group(1)) if match else -1

# List and sort files
tfiles = sorted(os.listdir(folder_path), key=extract_number)

# Build a lookup dict {file_number: integrated_current}
file_data = {}
for f in tfiles:
    if not f.endswith(".txt"):
        continue
    file_num = extract_number(f)
    file_path = os.path.join(folder_path, f)

    try:
        df = pd.read_csv(file_path)

        if " Cell[uAs]" in df.columns:
            # get the last valid (non-NaN) value
            integrated_current = df[" Cell[uAs]"].dropna().iloc[-1]
        else:
            integrated_current = None

        file_data[file_num] = integrated_current
    except Exception as e:
        print(f"Error reading {f}: {e}")

# Update main table
main_table["Integrated current raw [uAs]"] = main_table["DIC file number"].map(file_data)

# Calculate Raw DIC
def calc_dic(row):
    if pd.notna(row["Integrated current raw [uAs]"]) and pd.notna(row["Volume (mL)"]):
        return float(row["Integrated current raw [uAs]"] / ((row["Volume (mL)"]/1000) * FARADAY))
    else:
        return None

main_table["Raw DIC (umol/L)"] = main_table.apply(calc_dic, axis=1)

# Save updated table
main_table.to_csv(out_csv, index=False)

print(f"✅ Updated table saved to {out_csv}")
