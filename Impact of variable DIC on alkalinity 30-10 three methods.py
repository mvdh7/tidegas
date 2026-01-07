# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:31:57 2025

@author: nicor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
import calkulate as calk
import PyCO2SYS as pyco2
from scipy.optimize import curve_fit

# ---------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
})

# ---------------------------------------------------------------
# Import database and logbook
# ---------------------------------------------------------------
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

excel_file = "logbook_automated_by_python_testing.xlsx"
excel_df = pd.read_excel(excel_file)

# Filter by date
day, month = 11, 11
day, month = 30, 10
dbs = dbs[(dbs.analysis_datetime.dt.month == month) & (dbs.analysis_datetime.dt.day == day)]
dbs["titrant_molinity"] = excel_df["Titrant Molinity"]
dbs["salinity"] = excel_df["Salinity"]
dbs["temperature_override"] = excel_df["Temperature"]
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]

# Density fix
dbs["analyte_mass"] = (
    1e-3 * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)

# ---------------------------------------------------------------
# Load Excel file and clean data
# ---------------------------------------------------------------
df = pd.read_excel(excel_file)

required_cols = ["Calculated DIC (umol/kg)", "acid added (mL)", "date"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("Missing required columns in logbook Excel.")

plot_df = df.dropna(subset=["Calculated DIC (umol/kg)", "acid added (mL)", 
                            "date", "waiting time (minutes)"]).copy()

plot_df = plot_df[plot_df["waiting time (minutes)"] <= 0.05]
plot_df = plot_df[plot_df["acid increments (mL)"] <= 0.15]
plot_df = plot_df[plot_df["date"] == "30/10/2025"]
plot_df = plot_df[plot_df["bottle"] != "1"]

# plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")
# plot_df = plot_df[plot_df["Date"] >= "2025-11-10"]

plot_df["Titrant Volume (ml)"] = pd.to_numeric(plot_df["acid added (mL)"])
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"])
plot_df["Calculated in situ DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"])

plot_df["DIC (%)"] = 100 * plot_df["Calculated DIC (umol/kg)"] / plot_df["Reference DIC (umol/kg)"]
plot_df["In-situ DIC (%)"] = 100 * plot_df["Calculated in situ DIC (umol/kg)"] / plot_df["Reference DIC (umol/kg)"]

ref_dic = plot_df["Reference DIC (umol/kg)"].iloc[0]

# ---------------------------------------------------------------
# Polynomial fit for in-situ DIC
# ---------------------------------------------------------------
fit_df = plot_df.sort_values(by="Titrant Volume (ml)")

x = fit_df["Titrant Volume (ml)"].values
y = fit_df["In-situ DIC (%)"].values

coeffs = np.polyfit(x[:-2], y[:-2], 4)
poly = np.poly1d(coeffs)

x_new = np.linspace(0, 4.05, 28)
y_fit = poly(x_new)

fit_df_out = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": (y_fit / 100) * ref_dic
})

# ---------------------------------------------------------------
# Prepare titration object
# ---------------------------------------------------------------
tt = calk.to_Titration(dbs, 200)
ttt = tt.titration

base_totals = {k: ttt[k].values for k in ttt.columns if k.startswith("total_") or k == "dic"}

k_constants = {
    k: ttt[k].values
    for k in [
        "k_alpha", "k_ammonia", "k_beta", "k_bisulfate", "k_borate",
        "k_carbonic_1", "k_carbonic_2", "k_fluoride",
        "k_phosphoric_1", "k_phosphoric_2", "k_phosphoric_3",
        "k_silicate", "k_sulfide", "k_water",
    ]
}

# ---------------------------------------------------------------
# DIC variants for looping

# ---------------------------------------------------------------
dic_variants = {
    "DIC-Model with offset":  np.array((fit_df_out["Absolute DIC (umol/kg)"] + 127.46661137468027) * 1e-6),
    "DIC-Model no offset":    np.array((fit_df_out["Absolute DIC (umol/kg)"]) * 1e-6),
    "constant DIC":     np.ones(len(ttt.titrant_mass.values)) * ref_dic * 1e-6
}

variant_results = {}

# ---------------------------------------------------------------
# Loop over variants (ALKALINITY-FOCUSED VERSION)
# ---------------------------------------------------------------
for name, dic_array in dic_variants.items():

    totals_var = base_totals.copy()
    totals_var["dic"] = dic_array

    # --- Solve alkalinity for this DIC variant ---
    sr_var = calk.core.solve_emf(
        tt.titrant_molinity,
        ttt.titrant_mass.values,
        ttt.emf.values,
        ttt.temperature.values,
        tt.analyte_mass,
        totals_var,
        k_constants,
    )

    # Store alkalinity evolution through the titration
    df_var = pd.DataFrame({
        "titrant_mass": ttt.titrant_mass.values,
        "Titrant Volume (ml)": np.linspace(0,4.05,len(ttt.titrant_mass)),
        "alkalinity_all": sr_var.alkalinity_all,
        "alkalinity_final": sr_var.alkalinity,
        "variant": name
    })

    variant_results[name] = df_var

#%%
# ---------------------------------------------------------------
# Plot alkalinity for all DIC variants
# ---------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.grid(True, alpha=0.4)
ax = plt.gca()
for name, dfv in variant_results.items():
    plt.scatter(
        dfv["Titrant Volume (ml)"], 
        dfv["alkalinity_all"],
        label=name
    )
# ---- Manual shading: indices 17 to 24 ----
shade_start  =dfv["Titrant Volume (ml)"].iloc[17]
shade_end   = dfv["Titrant Volume (ml)"].iloc[24]

ax.axvspan(shade_start, shade_end, color="gray", alpha=0.25, label="Used region")

plt.xlabel("Titrant volume (ml)")
plt.ylabel("Alkalinity (µmol/kg)")
#plt.title("Alkalinity vs. Titrant Mass for Different DIC Variants")
plt.legend()
plt.tight_layout()
plt.show()

