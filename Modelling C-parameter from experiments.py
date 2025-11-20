# -*- coding: utf-8 -*-
"""
Extract C-values from exponential fits,
then overlay them as C(%) vs acid increment (mL)
onto the theoretical DIC figure from the second script.
"""

# ============================================================
# ====================== FIRST PART ==========================
# ===== Extract C(%) values from logbook exponential fits =====
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------- STYLE --------
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})

# -------- CONFIG --------
main_file = "Logbook_automated_by_python_testing.xlsx"
date_col = "date"
waiting_col = "waiting time (minutes)"
dic_col = "Calculated DIC (umol/kg)"
ref_dic_col = "Reference DIC (umol/kg)"
acid_col = "acid increments (mL)"      # incremental acid volume
acid_total_col = "acid added (mL)"     # total acid added per run

acid_totals_to_plot = [0.6, 1.2, 1.65, 2.1, 3, 4.2]
selected_dates = None
do_regression_lines = True

# -------- Load & preprocess --------
df = pd.read_excel(main_file)
print(f"Loaded {len(df)} rows from {main_file}")

# Ensure numeric columns
for col in [dic_col, ref_dic_col, waiting_col, acid_col, acid_total_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute relative DIC (%)
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

# Exponential decay model
def exp_decay(t, t0, C, k):
    return (100 - C) * np.exp(-k * (t - t0)) + C

# ===== STORAGE for all fitted C parameters =====
fit_results = []

# -------- Loop through acid totals --------
for acid_total in acid_totals_to_plot:

    df_sub = df[df[acid_total_col] == acid_total].dropna(
        subset=[waiting_col, "Relative DIC (%)", acid_col, date_col]
    )

    if df_sub.empty:
        continue

    # GUI filter
    if selected_dates is not None:
        df_sub = df_sub[df_sub[date_col].isin(selected_dates)]

    groups = df_sub.groupby([date_col, acid_col])

    # Loop through each date–increment pair
    for (day, inc), group in groups:
        group = group.sort_values(waiting_col)
        x = group[waiting_col].values
        y = group["Relative DIC (%)"].values

        if len(x) < 6:
            continue  # need >=5 points to fit

        try:
            popt, _ = curve_fit(
                exp_decay,
                x, y,
                p0=[0, 95, 0.001],
                bounds=([min(x) - 10, 0, 0], [max(x) + 10, 120, 1]),
                maxfev=10000
            )

            t0, C, k = popt

            # ===== STORE C-VALUE =====
            fit_results.append({
                "date": day,
                "acid_increment_mL": inc,
                "acid_total_mL": acid_total,
                "C_percent": C
            })

            print(f"{day}, increment {inc} mL:  C = {C:.2f}%")

        except RuntimeError:
            print(f"Fit failed for {day}, inc={inc}")

# Convert collected results to DataFrame
fit_df = pd.DataFrame(fit_results)
print("\n\n===== C VALUES EXTRACTED =====")
print(fit_df)



# ============================================================
# ===================== SECOND PART ==========================
# ========== Your working theoretical DIC plot ===============
# ============================================================

# %%
import calkulate as calk
import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
import pandas as pd

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
excel_df = pd.read_excel(excel_file)


#set up day and month for filtering DBS and logbook file 
day = 11
month = 11
date = str(day)+"/"+str(month)+"/2025"


#%%
#update the dbs from the logbook, and specify a date
# TODO might as well make this smarter and use
dbs = dbs[(dbs.analysis_datetime.dt.month == month) & (dbs.analysis_datetime.dt.day == day)]
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30
dbs["temperature_override"] = excel_df["Temperature"]  # Uses the temperature from the logbook
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]
# dbs["total_phosphate"] = 10
# dbs['total_silicate'] = 100
# TODO Workaround for density storing bug in v23.7.0
dbs["analyte_mass"] = (
    1e-3
    * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)


#%%
# Make sure columns exist
if not all(col in excel_df.columns for col in ["Calculated DIC (umol/kg)", "acid added (mL)", "date"]):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = excel_df.dropna(subset=["Calculated DIC (umol/kg)", "acid added (mL)", "date", "waiting time (minutes)"]).copy()

#select only the files without any (significant) waiting time 
#plot_df =plot_df[plot_df["waiting time (minutes)"]<=0.05]
plot_df =plot_df[plot_df["acid increments (mL)"]<=0.15]
plot_df = plot_df[plot_df["date"]==date]
plot_df = plot_df[plot_df["Mixing and waiting time (seconds)"]==4]
#plot_df = plot_df[plot_df["batch"]==1]

plot_df["acid added (mL)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Percentage DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')



#get one titration
tt = calk.to_Titration(dbs, 275)
ttt = tt.titration

totals = {k: ttt[k].values for k in ttt.columns if k.startswith("total_") or k == "dic"}

# totals["dic"] *= 0
# ^ make a numpy array (NOT pandas series) that is the same shape as
# ttt.titrant_mass.values that contains whatever DIC should be!

k_constants = {
    k: ttt[k].values
    for k in [
        "k_alpha",
        "k_ammonia",
        "k_beta",
        "k_bisulfate",
        "k_borate",
        "k_carbonic_1",
        "k_carbonic_2",
        "k_fluoride",
        "k_phosphoric_1",
        "k_phosphoric_2",
        "k_phosphoric_3",
        "k_silicate",
        "k_sulfide",
        "k_water",
    ]
}
sr = calk.core.solve_emf(
    tt.titrant_molinity,
    ttt.titrant_mass.values,
    ttt.emf.values,
    ttt.temperature.values,
    tt.analyte_mass,
    totals,
    k_constants,
    # alkalinity_init=None,
    # double=True,
    # emf0_init=None,
    # gran_logic="v23.7+",
    # pH_min=3,
    # pH_max=4,
    # titrant_normality=1,
)

# Calculate expected DIC based on alkalinity and pH
alkalinity_mixture = (
    sr.alkalinity * tt.analyte_mass - 1e6 * tt.titrant_molinity * ttt.titrant_mass
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

# What should the C value be?
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

fig, ax = plt.subplots(dpi=250)
ax.scatter(ttt.titrant_mass * 1e3, 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0],label = "Equilibrium C expected from theory")
# Plot fitted C values (simple scatter)
ax.scatter(
    fit_df["acid_total_mL"], 
    fit_df["C_percent"],
    color="red",
    s=60,
    label="Measured C(%) from exponential fits"
)

ax.set_xlabel("Titrant added (mg)  OR  Acid increment (mL)")
ax.set_ylabel("Relative DIC (%)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
