# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:13:18 2026

@author: nicor
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
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
# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
C_factors = np.linspace(0.95, 1.05, 50)
k0 = k  # s^-1

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for C sensitivity analysis.")
else:
    acid_levels = np.sort(sens_df["acid added (mL)"].unique())

    slopes_C = []
    acids = []

    for acid in acid_levels:

        sub = sens_df[sens_df["acid added (mL)"] == acid]

        all_C = []
        all_delta = []

        for _, row in sub.iterrows():

            C0 = row["C_from_reference (%)"]
            D_meas = row["Percentage DIC (%)"]

            C_test = C_factors * C0

            D0_pct = (
                C_test
                + (D_meas - C_test) * np.exp(k0 * 95)
            )

            D0_ref = (
                C0
                + (D_meas - C0) * np.exp(k0 * 95)
            )

            delta = D0_pct - D0_ref

            all_C.extend(C_test)
            all_delta.extend(delta)

        slope, intercept, r, _, _ = linregress(all_C, all_delta)

        acids.append(acid)
        slopes_C.append(slope)

    acids = np.array(acids)
    slopes_C = np.array(slopes_C)

    # ---------------------------------------------------------
    # PLOT: SLOPE VS ACID
    # ---------------------------------------------------------
    plt.figure()
    plt.grid(alpha=0.3)
    plt.scatter(
        acids,
        slopes_C,
        s=50,
        alpha=0.8
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Titrant Volume (mL)")
    plt.ylabel(r"Sensitivity slope  $\partial(\Delta \mathrm{DIC})/\partial C$")

    
    plt.tight_layout()
    plt.show()
#%%
# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
C_factors = np.linspace(0.95, 1.05, 50)
k0 = k  # s^-1

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for C sensitivity analysis.")
else:
    plt.figure()
    plt.grid(alpha=0.5)
    x_all = []
    y_all = []
    acid_all = []

    for _, row in sens_df.iterrows():

        C0 = row["C_from_reference (%)"]
        D_meas = row["Percentage DIC (%)"]

        # Perturbed C
        C_test = C_factors * C0

        # In-situ DIC with perturbed C
        D0_test = (
            C_test
            + (D_meas - C_test) * np.exp(k0 * 95)
        )

        # Reference in-situ DIC
        D0_ref = (
            C0
            + (D_meas - C0) * np.exp(k0 * 95)
        )

        # Relative changes (%)
        x_rel = 100 * C_test / C0
        y_rel = 100 * (D0_test - D0_ref) / D0_ref

        x_all.extend(x_rel)
        y_all.extend(y_rel)
        acid_all.extend([row["acid added (mL)"]] * len(x_rel))

    sc = plt.scatter(
        x_all,
        y_all,
        c=acid_all,
        cmap="viridis",
        s=25,
        alpha=0.7
    )

    # ---------------------------------------------------------
    # DECORATION
    # ---------------------------------------------------------
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(100, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Relative change in C (%)")
    plt.ylabel("Relative change in in situ DIC (%)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Titrant Volume (mL)")

   
    plt.tight_layout()
    plt.show()


#%%

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
C_factors = np.linspace(0.95, 1.05, 50)
k0 = k  # s^-1

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for C sensitivity analysis.")
else:
    acids = np.sort(sens_df["acid added (mL)"].unique())

    slopes = []

    # ---------------------------------------------------------
    # LOOP OVER UNIQUE ACID VALUES
    # ---------------------------------------------------------
    for acid in acids:

        sub = sens_df[sens_df["acid added (mL)"] == acid]

        x_rel = []
        y_rel = []

        for _, row in sub.iterrows():

            C0 = row["C_from_reference (%)"]
            D_meas = row["Percentage DIC (%)"]

            C_test = C_factors * C0

            D0_test = (
                C_test
                + (D_meas - C_test) * np.exp(k0 * 95)
            )

            D0_ref = (
                C0
                + (D_meas - C0) * np.exp(k0 * 95)
            )

            # Normalized changes (%)
            x_rel.extend(100 * C_test / C0)
            y_rel.extend(100 * (D0_test - D0_ref) / D0_ref)

        x_rel = np.array(x_rel)
        y_rel = np.array(y_rel)

        # Linear sensitivity slope
        slope, _, _, _, _ = linregress(x_rel, y_rel)
        slopes.append(slope)

    slopes = np.array(slopes)

    # ---------------------------------------------------------
    # PLOT: SLOPE VS ACID
    # ---------------------------------------------------------
    plt.figure()
    plt.grid(alpha=0.5)
    plt.scatter(acids, slopes, s=60)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Acid added (mL)")
    plt.ylabel(r"Sensitivity slope  $\Delta \mathrm{DIC}\% / \Delta C\%$")
   
    
    plt.tight_layout()
    plt.show()
#%%
# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
C_factors = np.linspace(0.95, 1.05, 50)
k0 = k  # s^-1

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for C sensitivity analysis.")
else:
    acids = np.sort(sens_df["acid added (mL)"].unique())

    mean_slopes = []
    std_slopes = []

    # ---------------------------------------------------------
    # LOOP OVER UNIQUE ACID VALUES
    # ---------------------------------------------------------
    for acid in acids:

        sub = sens_df[sens_df["acid added (mL)"] == acid]

        slopes = []

        for _, row in sub.iterrows():

            C0 = row["C_from_reference (%)"]
            D_meas = row["Percentage DIC (%)"]

            C_test = C_factors * C0

            D0_test = (
                C_test
                + (D_meas - C_test) * np.exp(k0 * 95)
            )

            D0_ref = (
                C0
                + (D_meas - C0) * np.exp(k0 * 95)
            )

            x_rel = 100 * C_test / C0
            y_rel = 100 * (D0_test - D0_ref) / D0_ref

            slope, _, _, _, _ = linregress(x_rel, y_rel)
            slopes.append(slope)

        slopes = np.array(slopes)
        mean_slopes.append(slopes.mean())
        std_slopes.append(slopes.std())

    mean_slopes = np.array(mean_slopes)
    std_slopes = np.array(std_slopes)

    # ---------------------------------------------------------
    # PLOT: MEAN SLOPE VS ACID WITH ERROR BARS
    # ---------------------------------------------------------
    plt.figure()
    plt.grid(alpha=0.5)
    plt.errorbar(
        acids,
        mean_slopes,
        yerr=std_slopes,
        fmt='o',
        ecolor='gray',
        capsize=3,
        markersize=6,
        markerfacecolor='blue',
        alpha=0.8
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Acid added (mL)")
    plt.ylabel(r"Sensitivity slope  $\Delta \mathrm{DIC}\% / \Delta C\%$")
  

    
    plt.tight_layout()
    plt.show()
#%%import numpy as np
# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------
C_factors = np.linspace(0.95, 1.05, 50)
k0 = k  # s^-1

mask = (
    (log["waiting time (minutes)"] == 0) &
    (~log["C_from_reference (%)"].isna()) &
    (~log["Percentage DIC (%)"].isna()) &
    (~log["acid added (mL)"].isna())
)

sens_df = log.loc[mask].copy()

if sens_df.empty:
    print("⚠ No rows available for C sensitivity analysis.")
else:
    acids = np.sort(sens_df["acid added (mL)"].unique())

    mean_slopes = []
    std_slopes = []
    mean_C0 = []

    # ---------------------------------------------------------
    # LOOP OVER UNIQUE ACID VALUES
    # ---------------------------------------------------------
    for acid in acids:

        sub = sens_df[sens_df["acid added (mL)"] == acid]

        slopes = []
        C0_values = []

        for _, row in sub.iterrows():

            C0 = row["C_from_reference (%)"]
            D_meas = row["Percentage DIC (%)"]
            C0_values.append(C0)

            C_test = C_factors * C0

            D0_test = (
                C_test
                + (D_meas - C_test) * np.exp(k0 * 95)
            )

            D0_ref = (
                C0
                + (D_meas - C0) * np.exp(k0 * 95)
            )

            x_rel = 100 * C_test / C0
            y_rel = 100 * (D0_test - D0_ref) / D0_ref

            slope, _, _, _, _ = linregress(x_rel, y_rel)
            slopes.append(slope)

        slopes = np.array(slopes)
        mean_slopes.append(slopes.mean())
        std_slopes.append(slopes.std())
        mean_C0.append(np.mean(C0_values))

    mean_slopes = np.array(mean_slopes)
    std_slopes = np.array(std_slopes)
    mean_C0 = np.array(mean_C0)

    # ---------------------------------------------------------
    # PLOT: Mean slope vs acid with error bars
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots()

    ax1.errorbar(
        acids,
        mean_slopes,
        yerr=std_slopes,
        fmt='o',
        ecolor='gray',
        capsize=3,
        markersize=6,
        markerfacecolor='blue',
        alpha=0.8,
        label='ΔDIC vs ΔC slope'
    )
    ax1.grid(alpha=0.5)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Titrant Volume (mL)")
    ax1.set_ylabel(r"Sensitivity slope  $\Delta \mathrm{DIC}\% / \Delta C\%$")
   

    # ---------------------------------------------------------
    # Overlay C0 values (inverted y-axis)
    # ---------------------------------------------------------
    ax2 = ax1.twinx()
    ax2.scatter(acids, mean_C0, marker = "x", color = 'red', alpha=0.5,label=r'Mean $C_0$')
    ax2.set_ylabel(r"$C_0$ (%)", color='red')
    ax2.invert_yaxis()  # invert to show higher sensitivity at lower C0
    ax2.tick_params(axis='y', colors='red')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


    fig.tight_layout()
    plt.show()
