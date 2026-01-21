# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:22:53 2025

@author: nicor
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:15:35 2025

@author: nicor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calkulate as calk
import PyCO2SYS as pyco2

# -------- CONFIG --------
excel_file = "Logbook_automated_by_python_testing.xlsx"
dbs_file = "data/vindta/r2co2/Nico.dbs"
file_path = "data/vindta/r2co2/Nico"

# Criteria
target_acid_added = 4.2
target_acid_increment = 0.15
target_mixing = 4

# -------- LOAD DATA --------
excel_df = pd.read_excel(excel_file)
dbs = calk.read_dbs(dbs_file, file_path=file_path)

# -------- ADD STABLE INDEX --------
dbs["titration_index"] = dbs.index.copy()
excel_df["titration_index"] = excel_df.index.copy()
dbs["titrant_molinity"] = excel_df["Titrant Molinity"].values
dbs["salinity"] = excel_df["Salinity"].values
dbs["temperature_override"] = excel_df["Temperature"].values
dbs["dic"] = excel_df["Reference DIC (umol/kg)"].values
dbs["analyte_mass"] = (
    1e-3
    * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)

# -------- FILTER EXCEL & DBS FOR YOUR CRITERIA --------
plot_df = excel_df.dropna(subset=["Calculated DIC (umol/kg)", 
                                   "acid added (mL)", 
                                   "acid increments (mL)", 
                                   "date", 
                                   "waiting time (minutes)"]).copy()


plot_df = plot_df[
    (plot_df["acid added (mL)"] == target_acid_added) &
    (plot_df["acid increments (mL)"] == target_acid_increment) &
    (plot_df["Mixing and waiting time (seconds)"] == target_mixing)
].copy()

# Filter dbs to only titrations matching plot_df
dbs_filtered = dbs[dbs["titration_index"].isin(plot_df["titration_index"])].copy()

# -------- UPDATE DBS WITH EXCEL VALUES --------
dbs_filtered["titrant_molinity"] = plot_df["Titrant Molinity"].values
dbs_filtered["salinity"] = plot_df["Salinity"].values
dbs_filtered["temperature_override"] = plot_df["Temperature"].values
dbs_filtered["dic"] = plot_df["Reference DIC (umol/kg)"].values


# Density workaround
dbs_filtered["analyte_mass"] = (
    1e-3 * dbs_filtered.analyte_volume / 
    calk.density.seawater_1atm_MP81(
        temperature=dbs_filtered.temperature_override,
        salinity=dbs_filtered.salinity
    )
)

# -------- PROCESS TITRATIONS --------
results = []

for tid in dbs_filtered["titration_index"]:
    try:
        tt = calk.to_Titration(dbs, int(tid))
        ttt = tt.titration
        print(tid)
        # --- totals (dic + totals) ---
        totals = {k: ttt[k].values for k in ttt.columns
                  if k.startswith("total_") or k == "dic"}

        # --- titrant volume ---
        ttt["titrant_volume"] = np.linspace(0, 4.05, num=len(ttt.titrant_mass.values))

        # --- K constants ---
        k_constants = {
            k: ttt[k].values
            for k in ["k_alpha","k_ammonia","k_beta","k_bisulfate","k_borate",
                      "k_carbonic_1","k_carbonic_2","k_fluoride",
                      "k_phosphoric_1","k_phosphoric_2","k_phosphoric_3",
                      "k_silicate","k_sulfide","k_water"]
        }

        # --- solve EMF ---
        sr = calk.core.solve_emf(
            tt.titrant_molinity,
            ttt.titrant_mass.values,
            ttt.emf.values,
            ttt.temperature.values,
            tt.analyte_mass,
            totals,
            k_constants
        )

        # --- alkalinity mixture ---
        alkalinity_mixture = (
            sr.alkalinity * tt.analyte_mass
            - 1e6 * tt.titrant_molinity * ttt.titrant_mass
        ) / (tt.analyte_mass + ttt.titrant_mass)
        
        #Theoretical DIC for alkalinity and pH
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
        # --- theoretical equilibrium C% ---
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

        c_percent = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]
        
        # --- Linear fit using first 15 points ---
        x_lin = ttt["titrant_volume"][:15]
        x_lin = ttt.titrant_mass.values[:15]
        y_lin = c_percent[:15]
        m, b = np.polyfit(x_lin, y_lin, 1)
        y_pred = m * x_lin + b
        
        # R^2
        ss_res = np.sum((y_lin - y_pred)**2)
        ss_tot = np.sum((y_lin - np.mean(y_lin))**2)
        r_squared = 1 - ss_res/ss_tot
        
        print(f"Linear fit: slope = {m:.3f}, intercept = {b:.3f}, R² = {r_squared:.3f}")

        plateau_c = float(np.mean(c_percent[-10:]))
        spread_c = float(np.std(c_percent[-10:]))
        
        # Store results
        results.append({
            "titration_index": tid,
            "Titrant Mass" : ttt.titrant_mass.values,
            "sample type" : plot_df.loc[plot_df["titration_index"]==tid, "sample/junk"].values[0],
            "file_not_good" : plot_df.loc[plot_df["titration_index"]==tid, "file_not_good"].values[0],
            "date": plot_df.loc[plot_df["titration_index"]==tid, "date"].values[0],
            "bottle nr": plot_df.loc[plot_df["titration_index"]==tid, "bottle"].values[0],
            "acid_added_mL": plot_df.loc[plot_df["titration_index"]==tid, "acid added (mL)"].values[0],
            "waiting time (minutes)": plot_df.loc[plot_df["titration_index"]==tid, "waiting time (minutes)"].values[0],
            "Titration duration (seconds)": plot_df.loc[plot_df["titration_index"]==tid, "Titration duration (seconds)"].values[0],
            "Titrant Molinity": plot_df.loc[plot_df["titration_index"]==tid, "Titrant Molinity"].values[0],
            "acid_increment_mL": plot_df.loc[plot_df["titration_index"]==tid, "acid increments (mL)"].values[0],
            "reference_DIC": plot_df.loc[plot_df["titration_index"]==tid, "Reference DIC (umol/kg)"].values[0],
            "calculated_DIC": plot_df.loc[plot_df["titration_index"]==tid, "Calculated DIC (umol/kg)"].values[0],
            "Alkalinity" : plot_df.loc[plot_df["titration_index"]==tid, "calkulate alkalinity"].values[0],
            "slope" : m,
            "intercept":b,
            "r_squared": r_squared,
            "plateau_C_percent": plateau_c,
            "spread_C_percent": spread_c,
            "Predicted 0 acid DIC from PYCO2SYS": co2s["dic"][0]
        })

    except Exception as e:
        print(f"Titration {tid} failed: {e}")

# -------- CREATE RESULTS DATAFRAME --------
results_df = pd.DataFrame(results)
# -------- APPEND plateau_C and predicted_DIC TO ORIGINAL EXCEL --------

# First create empty columns in the original df
excel_df["plateau_C_percent"] = np.nan
excel_df["Predicted_0acid_DIC"] = np.nan

# Now fill only the rows we processed
for _, row in results_df.iterrows():
    tid = row["titration_index"]
    excel_df.loc[excel_df["Titration index"] == tid, "plateau_C_percent"] = row["plateau_C_percent"]
    excel_df.loc[excel_df["Titration index"] == tid, "Predicted_0acid_DIC"] = row["Predicted 0 acid DIC from PYCO2SYS"]

# -------- SAVE A NEW EXCEL FILE --------
output_file = "logbook_with_plateau_and_predictedDIC.xlsx"
excel_df.to_excel(output_file, index=False)

print(f"Updated Excel saved to: {output_file}")

#%%
# -------- OPTIONAL: SAVE --------
#results_df.to_csv("titration_plateau_results_only_full_titrations.csv", index=False)
results_df.to_csv("titration_plateau_results_only_full_titrations_titrant_mass.csv", index=False)
#%%
plt.figure()
plt.scatter(results_df["sample type"],results_df["Predicted 0 acid DIC from PYCO2SYS"]-results_df["reference_DIC"])
plt.xlabel("Sample type")
plt.ylabel("DIC-offset (umol/kg)")