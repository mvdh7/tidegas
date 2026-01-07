# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:13:18 2026

@author: nicor
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import calkulate as calk
import PyCO2SYS as pyco2
import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})

# -------------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------------
file_path = "data/vindta/r2co2/Nico"
dbs_file = f"{file_path}.dbs"
excel_file = "logbook_automated_by_python_testing.xlsx"
output_filename = "logbook_in_situ_corrected2.xlsx"

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
k_min = 0.0294            # min^-1
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

#log = log[log["date"]!= "2025-11-10 00:00:00"] #skip the day where the bobometer was broken 
# have to shift the rows in the logbook manually, otherwise fine. 

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

        # Interpolated %C
        C = float(C_interp(acid))
        log.loc[idx, "C_from_reference (%)"] = C

        # Back-calc in situ DIC
        D0_pct = C + (D_meas - C) * np.exp(k * (wait_sec + 95))

        log.loc[idx, "Calculated in situ DIC (umol/kg)"] = D0_pct * (ref_DIC / 100)

#%%
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# SENSITIVITY ANALYSIS FOR WAITING TIME == 0
# -------------------------------------------------------------

# Nominal k (already defined earlier)
k0 = k  # s^-1

# ±10% range
k_values = np.linspace(0.9 * k0, 1.1 * k0, 50)

# Select rows where waiting time == 0 and calculation exists
mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["Reference DIC (umol/kg)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows with waiting time == 0 available for sensitivity analysis.")
else:
    dic_means = []

    for k_test in k_values:
        D0_pct = (
            sens_df["C_from_reference (%)"]
            + (sens_df["Percentage DIC (%)"] - sens_df["C_from_reference (%)"])
            * np.exp(k_test * 95)
        )

        D0_dic = D0_pct
        
        dic_means.append(D0_dic.mean())

    dic_means = np.array(dic_means)

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(k_values*60, dic_means)
    #plt.axvline(100, linestyle="--", linewidth=1)

    plt.xlabel("k-values")
    plt.ylabel("Mean in situ DIC (% of reference)")
    plt.tight_layout()
    plt.show()
#%%
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# SENSITIVITY ANALYSIS FOR WAITING TIME == 0
# -------------------------------------------------------------

k0 = k  # s^-1
k_values = np.linspace(0.9 * k0, 1.1 * k0, 50)

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (log["acid added (mL)"] >0.15)
)
# mask = (
#     (log["waiting time (minutes)"] == 0) &
#     (~log["C_from_reference (%)"].isna()) &
#     (~log["Percentage DIC (%)"].isna()) &
#     (log["acid added (mL)"] ==4.2)
# )
sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows with waiting time == 0 available for sensitivity analysis.")
else:
    plt.figure(figsize=(7, 5))

    for idx, row in sens_df.iterrows():

        D0_pct = (
            row["C_from_reference (%)"]
            + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
            * np.exp(k_values * 95)
        )
        D0_pct_ref = (
            row["C_from_reference (%)"]
            + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
            * np.exp(k0 * 95)
        )
        # Scatter (use plot for connected scatter)
        plt.grid(alpha =0.4)
        plt.plot(
            k_values * 60,
            D0_pct-D0_pct_ref,
            marker="o",
            markersize=3,
            alpha=0.6
        )

    plt.xlabel("k (min$^{-1}$)")
    plt.ylabel("In situ DIC (% of reference)")
    plt.tight_layout()
    plt.show()
#%%
from scipy.stats import linregress
import matplotlib.pyplot as plt

k0 = k  # s^-1
k_values = np.linspace(0.9 * k0, 1.1 * k0, 50)

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (log["acid added (mL)"] > 4.05)
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows with waiting time == 0 available for sensitivity analysis.")
else:
    plt.figure()

    all_k = []
    all_delta = []

    for _, row in sens_df.iterrows():

        D0_pct = (
            row["C_from_reference (%)"]
            + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
            * np.exp(k_values * 95)
        )

        D0_pct_ref = (
            row["C_from_reference (%)"]
            + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
            * np.exp(k0 * 95)
        )

        delta = D0_pct - D0_pct_ref

        plt.plot(
            k_values * 60,
            delta,
            marker="o",
            markersize=3,
            alpha=0.4,
            linewidth=0.8
        )

        all_k.extend(k_values * 60)
        all_delta.extend(delta)

    # ---------------------------------------------------------
    # FIT LINEAR SENSITIVITY
    # ---------------------------------------------------------
    slope, intercept, r, _, _ = linregress(all_k, all_delta)

    k_fit = np.array([k_values.min(), k_values.max()]) * 60
    delta_fit = slope * k_fit + intercept

    plt.plot(
        k_fit,
        delta_fit,
        color="black",
        linewidth=2,
        label="Linear sensitivity fit"
    )

    # ---------------------------------------------------------
    # PLOT DECORATION
    # ---------------------------------------------------------
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(k0 * 60, color="black", linestyle=":", linewidth=1)

    plt.grid(alpha=0.5)
    plt.xlabel("Gas exchange rate k (min$^{-1}$)")
    plt.ylabel("Δ in situ DIC (%)")


    # Sensitivity annotation
    text = (
        f"Sensitivity: {slope:.3f} % per min$^{{-1}}$\n"
        f"±10% k → ±{abs(slope * 0.1 * k0 * 60):.3f} % DIC\n"
        f"$R^2$ = {r**2:.3f}"
    )

    plt.text(
        0.02,
        0.97,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        fontsize = 14)
    

    plt.legend()
    plt.tight_layout()
    plt.show()
#%%
from scipy.stats import linregress
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
k0 = k  # s^-1
k_values = np.linspace(0.9 * k0, 1.1 * k0, 50)

# Select valid rows (waiting time = 0)
mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for sensitivity analysis.")
else:
    acid_levels = np.sort(sens_df["acid added (mL)"].unique())

    slopes = []
    acids = []

    # ---------------------------------------------------------
    # LOOP OVER ACID LEVELS
    # ---------------------------------------------------------
    for acid in acid_levels:

        sub = sens_df[sens_df["acid added (mL)"] == acid]

        all_k = []
        all_delta = []

        for _, row in sub.iterrows():

            D0_pct = (
                row["C_from_reference (%)"]
                + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
                * np.exp(k_values * 95)
            )

            D0_pct_ref = (
                row["C_from_reference (%)"]
                + (row["Percentage DIC (%)"] - row["C_from_reference (%)"])
                * np.exp(k0 * 95)
            )

            delta = D0_pct - D0_pct_ref

            all_k.extend(k_values * 60)   # convert to min^-1
            all_delta.extend(delta)

        # Linear fit: ΔDIC vs k
        slope, intercept, r, _, _ = linregress(all_k, all_delta)

        acids.append(acid)
        slopes.append(slope)

    acids = np.array(acids)
    slopes = np.array(slopes)

    # ---------------------------------------------------------
    # PLOT: SLOPE VS ACID
    # ---------------------------------------------------------
    plt.figure()
    plt.grid(alpha=0.5)
    plt.scatter(
        acids,
        slopes,
        s=50,
        alpha=0.8
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Titrant Volume (mL)")
    plt.ylabel(r"Sensitivity slope  $\partial(\Delta \mathrm{DIC})/\partial k$  (% min$^{-1}$)")

   
    plt.tight_layout()
    plt.show()

