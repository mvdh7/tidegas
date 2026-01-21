import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import calkulate as calk
import PyCO2SYS as pyco2

#calculated the average for daily reference measurements (indicated by "Alkalinity daily reference measurement" ==1 in the log)
# -------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------
file_path = "data/vindta/r2co2/Nico"
dbs_file = f"{file_path}.dbs"
excel_file = "logbook_automated_by_python_testing.xlsx"
output_filename = "logbook_alkalinity_corrected.xlsx"

# -------------------------------------------------------------
# LOAD DBS AND EXCEL LOGBOOK
# -------------------------------------------------------------
dbs = calk.read_dbs(dbs_file, file_path=file_path)
log = pd.read_excel(excel_file).copy()

# -------------------------------------------------------------
# UPDATE DBS FROM LOGBOOK
# -------------------------------------------------------------
dbs["titrant_molinity"] = log["Titrant Molinity"]
dbs["salinity"] = log["Salinity"]
dbs["temperature_override"] = log["Temperature"]
dbs["dic"] = log["Reference DIC (umol/kg)"]

# Workaround for density storing bug
dbs["analyte_mass"] = (
    1e-3
    * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)
#%%
# Load your logbook
excel_file = "logbook_automated_by_python_testing.xlsx"
log = pd.read_excel(excel_file).copy()

# Ensure date column is datetime
log["date"] = pd.to_datetime(log["date"], dayfirst=True, errors="coerce", format='%d/%m/%Y')

# Filter only reference measurements
ref_log = log[log["Alkalinity daily reference measurement"] == 1].copy()

# Group by date and sample type
daily_stats = (
    ref_log
    .groupby(["date", "sample/junk"])["calkulate alkalinity"]
    .agg(
        **{
            "daily reference alkalinity (umol/kg)": "mean",
            "daily reference alkalinity standard deviation (umol/kg)": "std"
        }
    )
    .reset_index()
)

# Initialize new columns with NaN
log["daily reference alkalinity (umol/kg)"] = np.nan
log["daily reference alkalinity standard deviation (umol/kg)"] = np.nan

# Assign the calculated values only to reference rows
for idx, row in daily_stats.iterrows():
    mask = (
        (log["date"] == row["date"]) &
        (log["sample/junk"] == row["sample/junk"]) &
        (log["Alkalinity daily reference measurement"] == 1)
    )
    log.loc[mask, "daily reference alkalinity (umol/kg)"] = row["daily reference alkalinity (umol/kg)"]
    log.loc[mask, "daily reference alkalinity standard deviation (umol/kg)"] = row["daily reference alkalinity standard deviation (umol/kg)"]

# Optional: view result
print(log[["date", "sample/junk", "Alkalinity daily reference measurement",
           "daily reference alkalinity (umol/kg)",
           "daily reference alkalinity standard deviation (umol/kg)"]])

# -------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------
#%%
log.to_excel(output_filename, index=False)
print(f"\nSAVED → {output_filename}")
