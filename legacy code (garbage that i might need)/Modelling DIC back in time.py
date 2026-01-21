# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:15:35 2025

"""

# %%
import calkulate as calk
import PyCO2SYS as pyco2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "figure.titlesize": 18,
})

def exp_decay(t,D0, C, k):
    return (D0-C) * np.exp(-k * (t)) + C

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
excel_df = pd.read_excel(excel_file)


#set up day and month for filtering DBS and logbook file 
day = 11
month = 11
# day = 30
# month = 10

date = str(day)+"/"+str(month)+"/2025"

#%%
#update the dbs from the logbook, and specify a date
dbs = dbs[(dbs.analysis_datetime.dt.month == month) & (dbs.analysis_datetime.dt.day == day)]
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30
dbs["temperature_override"] = excel_df["Temperature"]  # Uses the temperature from the logbook
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]
#dbs["total_phosphate"] = 1000
#dbs['total_silicate'] = 1000
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
# tt = calk.to_Titration(dbs, 200)
tt = calk.to_Titration(dbs, 275)
ttt = tt.titration

totals = {k: ttt[k].values for k in ttt.columns if k.startswith("total_") or k == "dic"}

# totals["dic"] *= 0
# ^ make a numpy array (NOT pandas series) that is the same shape as
# ttt.titrant_mass.values that contains whatever DIC should be!
ttt["titrant_volume"] = np.linspace(0, num = len(ttt.titrant_mass.values), stop = 4.05)

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

#%%

# --- SELECT ONE MEASUREMENT FOR FITTING ---
acid_ml = 0.6  # choose any value present in plot_df

# nearest plot_df row
row = plot_df.loc[plot_df["acid added (mL)"].sub(acid_ml).abs().idxmin()]
D0_pct = row["Percentage DIC (%)"]   # measured %DIC at t=0

# reference is always 100%
D_ref_pct = 100

# --- MATCH THEORETICAL C VALUE ---
vol_array = ttt["titrant_volume"]
C_array = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]   # convert to %DIC

idx = (vol_array - acid_ml).abs().argmin()
C_theory_pct = C_array[idx]   # equilibrium %DIC for this acid amount

# --- MODEL PARAM ---
k = 0.03

# --- FIND t_0 AT WHICH CURVE HITS 100% ---
t0 = - (1/k) * np.log((D_ref_pct - C_theory_pct) / (D0_pct - C_theory_pct))

# --- TIME AXIS ---
t = np.linspace(t0, 100, 400)   # from negative time to +10 min

# --- EXPONENTIAL CURVE ---
D_fit_pct = exp_decay(t, D0_pct, C_theory_pct, k)

# --- PLOT ---
plt.figure(figsize=(8,5))
plt.grid(True, zorder=0)

# fit curve
plt.plot(t, D_fit_pct, label="Exponential fit", linewidth=2)

# measured point
plt.scatter([0], [D0_pct], color='red', zorder=5, label="Measured %DIC at t=0")

# 100% line
plt.axhline(100, linestyle="--", color="gray", label="100% DIC")

plt.xlabel("Time (minutes)")
plt.ylabel("DIC (%)")
plt.title(f"Back-calculated DIC decay for {acid_ml} mL titrant")
plt.legend()
plt.tight_layout()
plt.show()

print("Initial time offset t0 (minutes):", t0)
#%%
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def find_times_for_acid(
        acid_ml,
        plot_df,
        ttt,
        co2s_fco2,
        k_min=0.03,
        plot=True
    ):
    """
    Computes the instantaneous and 60-second integrated time for a DIC measurement
    assuming an exponential decay starting at 100% at t=0.

    Returns a dictionary with times and values.
    """

    # --- extract measured DIC (%), nearest to acid_ml ---
    row = plot_df.loc[plot_df["acid added (mL)"].sub(acid_ml).abs().idxmin()]
    D_meas = row["Percentage DIC (%)"]

    # --- theoretical equilibrium C (%) ---
    vol_array = ttt["titrant_volume"]
    C_array = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]
    idx = (vol_array - acid_ml).abs().argmin()
    C = C_array[idx]

    # --- convert k to per second ---
    k = k_min / 60.0

    # exponential curve
    def D_curve(t):
        return (100 - C) * np.exp(-k * t) + C

    # integral of D(t)
    def integral_D(t):
        return -(100 - C)/k * np.exp(-k*t) + C * t

    # 60-second centered average
    def avg_centered(tcenter):
        t1, t2 = tcenter - 30, tcenter + 30
        return (integral_D(t2) - integral_D(t1)) / 60

    # ------------------------------------------------------
    # Solve instantaneous time: D(t_inst) = D_meas
    # ------------------------------------------------------
    t_inst = - (1/k) * np.log((D_meas - C) / (100 - C))

    # ------------------------------------------------------
    # Solve t_int such that centered average = D_meas
    # Use robust method to avoid brentq errors
    # ------------------------------------------------------
    try:
        t_int = brentq(lambda t: avg_centered(t) - D_meas, 0, 1200)
    except ValueError:
        # fallback: numerical search
        t_grid = np.linspace(0, 1200, 10000)
        avg_grid = np.array([avg_centered(tt) for tt in t_grid])
        t_int = t_grid[np.argmin(np.abs(avg_grid - D_meas))]

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    if plot:
        t_plot = np.linspace(0, 1200, 2400)
        Dvals = D_curve(t_plot)

        plt.figure(figsize=(10,6))
        plt.grid(True)

        plt.plot(t_plot, Dvals, label="Master decay curve (100% → C)", linewidth=2)

        # Instantaneous point
        plt.scatter([t_inst], [D_meas], color='red', s=80,
                    label=f"Instantaneous match (t={t_inst:.2f}s)")

        # Integrated point
        plt.scatter([t_int], [D_meas], color='green', s=80,
                    label=f"Integrated ±30s match (t={t_int:.2f}s)")

        # integration window
        plt.axvspan(t_int - 30, t_int + 30,
                    color='green', alpha=0.15,
                    label="±30s averaging window")

        plt.xlabel("Time (seconds)")
        plt.ylabel("DIC (%)")
        plt.title(f"Decay placement for {acid_ml} mL titrant\n"
                  f"Shift={t_int - t_inst:.1f} s")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "acid_ml": acid_ml,
        "D_meas": D_meas,
        "C": C,
        "k_sec": k,
        "t_inst": t_inst,
        "t_int": t_int,
        "time_shift": t_int - t_inst
    }
#%%
result = find_times_for_acid(
    acid_ml=3.3,
    plot_df=plot_df,
    ttt=ttt,
    co2s_fco2=co2s_fco2,
    k_min=0.03
)
#%%
print(result)
acids = sorted(plot_df["acid added (mL)"].unique())

results = []
for a in acids:
    out = find_times_for_acid(
        acid_ml=a,
        plot_df=plot_df,
        ttt=ttt,
        co2s_fco2=co2s_fco2,
        k_min=0.03,
        plot=False  # skip individual plots
    )
    results.append(out)

results_df = pd.DataFrame(results)

#%%
# Fixed time shift and uncertainty
delta_t = -95  # seconds
delta_t_err = 5  # ±5 s

# Exponential constant (per second)
k_min = 0.03
k = k_min / 60.0

# Prepare figure
plt.figure(figsize=(10,6))
plt.grid(True,alpha = 0.4)

# Sort acids for plotting
acids = sorted(plot_df["acid added (mL)"].unique())

original_pct = []
corrected_pct = []
corrected_upper = []
corrected_lower = []

for a in acids:
    # Measured %DIC
    row = plot_df.loc[plot_df["acid added (mL)"].sub(a).abs().idxmin()]
    D_meas = row["Percentage DIC (%)"]

    # Theoretical C
    idx = (ttt["titrant_volume"] - a).abs().argmin()
    C = 100 * co2s_fco2["dic"][idx] / co2s_fco2["dic"][0]

    # Corrected DIC with time shift
    D_corr = (D_meas - C) * np.exp(-k * delta_t) + C

    # Uncertainty band ± delta_t_err
    D_corr_upper = (D_meas - C) * np.exp(-k * (delta_t - delta_t_err)) + C
    D_corr_lower = (D_meas - C) * np.exp(-k * (delta_t + delta_t_err)) + C

    original_pct.append(D_meas)
    corrected_pct.append(D_corr)
    corrected_upper.append(D_corr_upper)
    corrected_lower.append(D_corr_lower)


# Plot original DIC
plt.plot(acids, np.array(original_pct), 'o-', color='red', label="Original %DIC")

# Plot corrected DIC
plt.plot(acids, np.array(corrected_pct), 'o-', color='blue', label=f"Corrected %DIC (Δt={delta_t}s)")
plt.scatter(ttt.titrant_mass*1000, 100*co2s["dic"]/co2s["dic"][0], label= "Expected DIC from PyCO2SYS")
# Plot uncertainty band
plt.fill_between(acids, corrected_lower, corrected_upper, color='blue', alpha=0.2,
                 label=f"±{delta_t_err}s uncertainty")

plt.xlabel("Acid added (mL)")
plt.ylabel("DIC (%)")
plt.title("Original vs Corrected DIC with Time Shift")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Fixed time shift and uncertainty
delta_t = -95  # seconds
delta_t_err = 5  # ±5 s

# Exponential constant (per second)
k_min = 0.03
k = k_min / 60.0

# Prepare figure

# Sort acids for plotting
acids = sorted(plot_df["acid added (mL)"].unique())

original_umol = []
corrected_umol = []
corrected_upper_umol = []
corrected_lower_umol = []

for a in acids:
    # Measured %DIC
    row = plot_df.loc[plot_df["acid added (mL)"].sub(a).abs().idxmin()]
    D_meas = row["Percentage DIC (%)"]
    pct_to_umol = plot_df["Reference DIC (umol/kg)"][275]/100
    # Theoretical C
    idx = (ttt["titrant_volume"] - a).abs().argmin()
    C = 100 * co2s_fco2["dic"][idx] / co2s_fco2["dic"][0]

    # Corrected DIC with time shift
    D_corr = (D_meas - C) * np.exp(-k * delta_t) + C

    # Uncertainty band ± delta_t_err
    D_corr_upper = (D_meas - C) * np.exp(-k * (delta_t - delta_t_err)) + C
    D_corr_lower = (D_meas - C) * np.exp(-k * (delta_t + delta_t_err)) + C

    original_umol.append(D_meas*pct_to_umol)
    corrected_umol.append(D_corr*pct_to_umol)
    corrected_upper.append(D_corr_upper*pct_to_umol)
    corrected_lower.append(D_corr_lower*pct_to_umol)

plt.figure(figsize=(10,6))
plt.grid(True,alpha =0.4)

# Plot original DIC
plt.plot(acids, np.array(original_umol), 'o-', color='red', label="Original %DIC")

# Plot corrected DIC
plt.plot(acids, np.array(corrected_umol), 'o-', color='blue', label=f"Corrected %DIC (Δt={delta_t}s)")
plt.plot(acids, np.array(corrected_umol)+co2s["dic"][0]-corrected_umol[0], 'o-', color='green', label=f"Corrected and shifted %DIC (Δt={delta_t}s)")
plt.scatter(ttt["titrant_volume"], co2s["dic"], label= "Expected DIC from PyCO2SYS")
# Plot uncertainty band
# plt.fill_between(np.array(acids), np.array(corrected_lower), np.array(corrected_upper), color='blue', alpha=0.2,
#                  label=f"±{delta_t_err}s uncertainty")
plt.ylim(1500,2400)
plt.xlabel("Acid added (mL)")
plt.ylabel(r"DIC ($\mu$mol/kg)")
plt.title("Original vs Corrected DIC with Time Shift")
plt.legend()
plt.tight_layout()
plt.show()
#%%


plt.figure(figsize=(10,6))
plt.grid(True,alpha =0.4)

# Plot original DIC
#plt.plot(acids, np.array(original_umol), 'o-', color='red', label="Original %DIC")

# Plot corrected DIC
plt.plot(acids, np.array(corrected_umol)-corrected_umol[0], 'o-', color='blue', label=f"Corrected %DIC (Δt={delta_t}s)")

plt.scatter(ttt["titrant_volume"], co2s["dic"]-co2s["dic"][0], label= "Expected DIC from PyCO2SYS")

# Plot uncertainty band
# plt.fill_between(np.array(acids), np.array(corrected_lower), np.array(corrected_upper), color='blue', alpha=0.2,
#                  label=f"±{delta_t_err}s uncertainty")
plt.ylim(-500,0)
plt.xlabel("Acid added (mL)")
plt.ylabel(r"DIC ($\mu$mol/kg)")
plt.title("Original vs Corrected DIC with Time Shift")
plt.legend()
plt.tight_layout()
plt.show()



