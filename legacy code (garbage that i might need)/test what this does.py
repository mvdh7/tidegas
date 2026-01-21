import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import calkulate as calk
import PyCO2SYS as pyco2

# -------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------
file_path = "data/vindta/r2co2/Nico"
dbs_file = f"{file_path}.dbs"
excel_file = "logbook_automated_by_python_testing.xlsx"
output_filename = "logbook_in_situ_corrected.xlsx"

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

# Ensure the date column is datetime
log["date"] = pd.to_datetime(log["date"], dayfirst=True, errors="coerce",format = '%d/%m/%Y')

# Dictionary to store first reference index per day
first_ref_indices = []
date_idx = []
# Loop over unique dates in sorted order
for date, group in log.groupby("date"):
    ref_rows = group[group["Alkalinity daily reference measurement"] == 1]
    if len(ref_rows) > 0:
        first_idx = ref_rows.index[0]
        first_ref_indices.append(first_idx)
        date_idx.append(date)

# Convert to array
#remove 10-11-2025 because the bobometer was broken there
date_idx.pop(21)
first_ref_indices.pop(21)
first_ref_indices = pd.Series(first_ref_indices).to_numpy()

print("First reference row index for each day (sorted by day/month/year):")
print(first_ref_indices)
print(date_idx)
#%%
# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
k_min = 0.03            # min^-1
k = k_min / 60          # convert to s^-1

# -------------------------------------------------------------
# PREPARE LOG FOR CALCULATION
# -------------------------------------------------------------
numeric_cols = ["acid added (mL)", "waiting time (minutes)",
                "Calculated DIC (umol/kg)", "Reference DIC (umol/kg)",
                "Alkalinity daily reference measurement"]

for col in numeric_cols:
    log[col] = pd.to_numeric(log[col], errors="coerce")

log["Percentage DIC (%)"] = 100 * log["Calculated DIC (umol/kg)"] / log["Reference DIC (umol/kg)"]

# Add empty columns for results
log["C_from_reference (%)"] = np.nan
log["Calculated in situ DIC (umol/kg)"] = np.nan
log["DIC-offset first point"] = np.nan
#log = log[log["date"]!= "2025-11-10 00:00:00"] #skip the day where the bobometer was broken 
# have to shift the rows in the logbook manually, otherwise fine. 
#%%

#%%
# -------------------------------------------------------------
# LOOP THROUGH DATES
# -------------------------------------------------------------
# Ensure date and reference indices are aligned
date_to_ref_idx = dict(zip(date_idx, first_ref_indices))

# -------------------------------------------------------------
# LOOP THROUGH DATES USING CORRECT DBS INDICES
# -------------------------------------------------------------
for date, group in log.groupby("date"):

    print(f"\n=== Processing date: {date} ===")
   
    # Check if we have a reference titration index for this date
    if date not in date_to_ref_idx:
        print(f"  ⚠ No reference titration found for {date}, skipping.")
        continue

    ref_idx = date_to_ref_idx[date]  # This is the row in the log/dbs to use
    print(f"  ✅ Using reference titration at DBS index {ref_idx}")

    # Use the correct DBS index for this date
    tt = calk.to_Titration(dbs, ref_idx)
    ttt = tt.titration
    
    totals = {k: ttt[k].values for k in ttt.columns
              if k.startswith("total_") or k == "dic"}

    ttt["titrant_volume"] = np.linspace(0, 4.05, num=len(ttt.titrant_mass.values))

    k_constants = {
        k: ttt[k].values for k in [
            "k_alpha","k_ammonia","k_beta","k_bisulfate","k_borate",
            "k_carbonic_1","k_carbonic_2","k_fluoride",
            "k_phosphoric_1","k_phosphoric_2","k_phosphoric_3",
            "k_silicate","k_sulfide","k_water"
        ]
    }

    # Solve alkalinity from EMF
    sr = calk.core.solve_emf(
        tt.titrant_molinity,
        ttt.titrant_mass.values,
        ttt.emf.values,
        ttt.temperature.values,
        tt.analyte_mass,
        totals,
        k_constants,
    )

    alkalinity_mixture = (
        sr.alkalinity * tt.analyte_mass
        - 1e6 * tt.titrant_molinity * ttt.titrant_mass
    ) / (tt.analyte_mass + ttt.titrant_mass)
    
    co2s = pyco2.sys(
        par1=alkalinity_mixture.values,
        par2=ttt.pH.values,
        par1_type=1,
        par2_type=3,
        temperature=ttt.temperature.values,
        salinity=tt.salinity,
        opt_pH_scale=3,
        uncertainty_from={"par1": 5, "par2": 0.01},
        uncertainty_into=["dic"],
    )

    co2s_fco2 = pyco2.sys(
        par1=alkalinity_mixture.values,
        par2=500,
        par1_type=1,
        par2_type=5,
        temperature=ttt.temperature.values,
        salinity=tt.salinity,
        uncertainty_from={"par1": 5, "par2": 50},
        uncertainty_into=["dic"],
    )

    # Convert to %C-equilibrium
    acid_ref = ttt["titrant_volume"].values
    C_ref = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]
    C_interp = interp1d(acid_ref, C_ref, bounds_error=False, fill_value="extrapolate")
    
    # Compare the difference between the reference measured and the theoretical first point
    
    # ---------------------------------------------------------
    # PROCESS ROWS WHERE CALCULATION IS POSSIBLE
    # ---------------------------------------------------------
    for idx, row in group.iterrows():
        if pd.isna(row["acid added (mL)"]) or pd.isna(row["Percentage DIC (%)"]) or pd.isna(row["waiting time (minutes)"]):
            continue  # Skip rows without necessary info

        acid = float(row["acid added (mL)"])
        D_meas = float(row["Percentage DIC (%)"])
        wait_sec = float(row["waiting time (minutes)"]) * 60
        ref_DIC = float(row["Reference DIC (umol/kg)"])
        
        #record the difference between the reference DIC and the first value from co2s, which predicts the DIC based on alkalinity and pH
        First_point_difference = co2s["dic"][0] -ref_DIC
        log.loc[idx, "DIC-offset first point"] = First_point_difference
        
        # Interpolated %C
        C = float(C_interp(acid))
        log.loc[idx, "C_from_reference (%)"] = C

        # Back-calc in situ DIC
        D0_pct = C + (D_meas - C) * np.exp(k * (wait_sec + 95))
        log.loc[idx, "Calculated in situ DIC (umol/kg)"] = D0_pct * (ref_DIC / 100)

# -------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------
#%%
log.to_excel(output_filename, index=False)
print(f"\nSAVED → {output_filename}")
