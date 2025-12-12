# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:15:35 2025

@author: nicor
"""

# -*- coding: utf-8 -*-
"""
Plot Relative DIC (%) vs waiting time (minutes)
Grouped by date & acid increment (mL) with >=3 measurements
Exponential decay fit instead of linear
Handles multiple acid totals automatically
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
# -------- STYLE --------
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
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
mixing_col = "Mixing and waiting time (seconds)"
acid_totals_to_plot = [0.6,1.2,1.65,2.1,3, 4.2]       # <--- list of total acid volumes to process
#acid_totals_to_plot = np.linspace(0,4.2,15).round(4)
selected_dates = None                  # None or list of specific dates to include
do_regression_lines = True


# -------- Load & preprocess --------
df = pd.read_excel(main_file)
print(f"Loaded {len(df)} rows from {main_file}")
# Sort by date then bottle
df['date_sort'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
df.sort_values(by=['date_sort'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=['date_sort'], inplace=True)
# Ensure numeric columns
for col in [dic_col, ref_dic_col, waiting_col, acid_col, acid_total_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = np.nan

# Compute relative DIC (%)
df["Relative DIC (%)"] = (df[dic_col] / df[ref_dic_col]) * 100

#sort out longer mixing time measurements

df = df[df["Mixing and waiting time (seconds)"]==4]
# Exponential decay model
def exp_decay(t,D0, C, k):
    return (D0-C) * np.exp(-k * (t)) + C

fit_results = []
# -------- Function to plot for one acid total --------
def plot_acid_total(df, acid_total, use_colorbar_mode=False):
    df_sub = df[df[acid_total_col] == acid_total].dropna(
        subset=[waiting_col, "Relative DIC (%)", acid_col, date_col]
    )
    if df_sub.empty:
        print(f"⚠️ No data found for acid total = {acid_total} mL.")
        return

    # Convert date to datetime
    df_sub["date_dt"] = pd.to_datetime(df_sub[date_col], format="%d/%m/%Y", errors="coerce")

    # Keep groups with ≥3 points
    group_counts = df_sub.groupby([date_col, acid_col]).size()
    valid_groups = group_counts[group_counts >= 1].index
    df_sub = df_sub.set_index([date_col, acid_col]).loc[valid_groups].reset_index()

    if selected_dates is not None:
        df_sub = df_sub[df_sub[date_col].isin(selected_dates)]

    if df_sub.empty:
        print(f"⚠️ No valid (date, increment) groups for {acid_total} mL.")
        return

    # Sort groups chronologically
    grouped = list(df_sub.groupby([date_col, acid_col]))
    grouped.sort(key=lambda g: pd.to_datetime(g[0][0], format="%d/%m/%Y"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, zorder=0)
    # For colorbar mode
    if use_colorbar_mode:
        cmap = plt.cm.plasma
        norm = plt.Normalize( mdates.date2num(df_sub["date_dt"].min()), mdates.date2num(df_sub["date_dt"].max()))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    # Store fit parameters
    param_lines = []

    colors = plt.cm.tab10.colors
    for i, ((day, inc), group) in enumerate(grouped):
        group = group.sort_values(waiting_col)
        x = group[waiting_col].values
        y = group["Relative DIC (%)"].values
        date_ts = mdates.date2num(group["date_dt"].iloc[0])


        if use_colorbar_mode:
            color = cmap(norm(date_ts))
            marker = "o" if abs(inc - 0.15) < 1e-6 else "x"
            label = None   # legend removed in colorbar mode
        else:
            color = colors[i % len(colors)]
            marker = "o"
            label = f"{day} — {inc:.2f} mL"
        
        ax.scatter(x, y, color=color, marker=marker, s=70, label=label)
        ax.errorbar(x,y,xerr=10/60, yerr = 1, color=color,marker = marker, capsize = 4, fmt= "none",alpha = 0.5)
        #ax.plot(x, y, color=color, alpha=0.6)

        # Fit
        if do_regression_lines and len(x) >= 5 and max(group[waiting_col])>=25:
            try:
                p0 = [80, 95, 0.001]
                bounds = ([50, 0, 0], [100, 120, 1])
                popt, pcov = curve_fit(
                    exp_decay, x, y,
                    p0=p0, bounds=bounds,
                    sigma=np.ones_like(y) * 1.0,  # 1% error, or any real y-error
                    absolute_sigma=True,
                    maxfev=10000)
                
                print(pcov)
                D0_err = np.sqrt(pcov[0,0])
                C_err = np.sqrt(pcov[1,1])
                k_err = np.sqrt(pcov[2, 2])

                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, exp_decay(xs, *popt), linestyle='--', color=color, alpha=0.7)

                # Collect fit parameters for side box
                D0, C, k = popt
                param_lines.append(f"{day}, incr={inc:.2f}:  k={k:.4f}, C={C:.1f}, D$_{{{0}}}$ = {D0:.1f}")
                D0, C, k = popt

                # ===== STORE C-VALUE =====
                fit_results.append({
                    "date": day,
                    "acid_increment_mL": inc,
                    "acid_total_mL": acid_total,
                    "C_percent": C,
                    "k": k,
                    "kerr": k_err,
                    "Cerr" :C_err,
                    "D0err" : D0_err, 
                    "D0": D0})
                
                    
                print(f"{day}, increment {inc} mL:  C = {C:.2f}%")

          
            except RuntimeError:
                fit_results.append(f"{day}, inc={inc:.2f}:  fit FAILED")

    # ------------------------- SIDE FIT-PARAMETER BOX -------------------------
    if param_lines:
        textstr = "\n".join(param_lines)
        ax.text(
            0.3, 0.70, textstr,
            transform=ax.transAxes,
            fontsize=14,
            va='center',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )
    textstr = r"$DIC(t) = (D_0-C)e^{-kt} + C$"
    ax.text(
        0.6, 0.9, textstr,
        transform=ax.transAxes,
        fontsize=14,
        va='center',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    # ------------------------- LABELS, COLORBAR, LEGEND -------------------------
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Relative DIC (% of reference)")
    ax.set_title(f"DIC decay — {acid_total:.2f} mL acid total")

    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    

    if use_colorbar_mode:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Date")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.legend(title="Day — Acid increment", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    plt.show()


# -------- Run for all acid totals --------
for acid_total in acid_totals_to_plot:
    plot_acid_total(df, acid_total, use_colorbar_mode=True)

fit_df = pd.DataFrame(fit_results)
print("\n\n===== C VALUES EXTRACTED =====")
print(fit_df)


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
# day = 30
# month = 10

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
# tt = calk.to_Titration(dbs, 200)
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

plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "figure.titlesize": 18,
})

plt.figure()
plt.grid(True, zorder=0)
#ax.scatter(ttt["titrant_volume"], 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0],label = "Equilibrium C% expected from theory")
# Plot fitted C values (simple scatter)
plt.scatter(
    fit_df["acid_total_mL"], 
    fit_df["C_percent"],
    color="red",
    s=80,
    marker= 'x',
    label=r"$C$(%) from exponential fits"
)
plt.errorbar(fit_df["acid_total_mL"], fit_df["C_percent"], yerr= fit_df["Cerr"],color="red",capsize = 3, fmt= "none")
plt.xlabel("Acid added (mL)")
plt.ylabel("C (%)")
plt.legend()
plt.tight_layout()
plt.show()
#%%

plt.figure()
plt.grid(True, zorder=0)
#ax.scatter(ttt["titrant_volume"], 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0],label = "Equilibrium C% expected from theory")
# Plot fitted C values (simple scatter)
plt.scatter(
    fit_df["acid_total_mL"], 
    fit_df["k"],
    color="red",
    s=80,
    marker= 'x',
    label=r"$k$ (1/min) from exponential fits"
)
plt.errorbar(fit_df["acid_total_mL"], fit_df["k"], yerr= fit_df["kerr"],color="red",capsize = 3, fmt= "none")
plt.xlabel("Acid added (mL)")
plt.ylabel("k (1/min)")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.grid(True, zorder=0)
#ax.scatter(ttt["titrant_volume"], 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0],label = "Equilibrium C% expected from theory")
# Plot fitted C values (simple scatter)
plt.scatter(
    fit_df["acid_total_mL"], 
    fit_df["D0"],
    color="red",
    s=80,
    marker= 'x',
    label="D0(%) from exponential fits"
)
plt.errorbar(fit_df["acid_total_mL"], fit_df["D0"], yerr= fit_df["D0err"],color="red",capsize = 3, fmt= "none")
plt.xlabel("Acid added (mL)")
plt.ylabel("Initial DIC (%)")
plt.legend()
plt.tight_layout()
plt.show()
#%%
def linear_jitter(x_values, max_offset=0.05):
    """
    Applies deterministic linear jitter within a group of identical x-values.
    Each unique x gets symmetrically spaced jitter offsets.
    """
    x_values = np.array(x_values)
    unique_vals = np.unique(x_values)
    jittered = np.zeros_like(x_values, dtype=float)

    for xv in unique_vals:
        idx = np.where(x_values == xv)[0]
        n = len(idx)

        if n == 1:
            jittered[idx] = xv
            continue

        # Linear offsets from -max_offset → +max_offset
        offsets = np.linspace(-max_offset, max_offset, n)
        jittered[idx] = xv + offsets

    return jittered
plt.figure(figsize=(8,5))
plt.grid(True, zorder=0)

acid = fit_df["acid_total_mL"].values
acid_j = linear_jitter(acid, max_offset=0.02)

plt.errorbar(
    acid_j,
    fit_df["C_percent"],
    yerr=fit_df["Cerr"],
    fmt="none",
    color="red",
    elinewidth=1.3,
    capsize=3,
    alpha=0.9,
    zorder=2
)

plt.scatter(
    acid_j,
    fit_df["C_percent"],
    color="red",
    s=60,
    marker="o",
    edgecolors="black",
    alpha=0.85,
    label="C (%) from fits",
    zorder=3
)

plt.xlabel("Acid added (mL)")
plt.ylabel("C (%)")
plt.legend()
plt.tight_layout()
plt.show()

#%%
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})
x = ttt["titrant_volume"]
y = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]
y_error = 100 * co2s_fco2["u_dic"] / co2s_fco2["dic"][0]

# --- Linear fit using first 15 points ---
x_lin = x[:15]
y_lin = y[:15]
m, b = np.polyfit(x_lin, y_lin, 1)
print(f"Linear fit: slope = {m:.3f}, intercept = {b:.3f}")

# --- Plateau: average of last 10 points ---
plateau = np.mean(y[-10:])
print(f"Plateau value (avg of last 10 points) = {plateau:.3f}")

# --- Piecewise ---
def linear_then_plateau(x):
    x_intercept = (plateau - b) / m
    return np.where(x < x_intercept, m*x + b, plateau)

# --- Figure structure ---
fig = plt.figure(figsize=(7,5), dpi=250)
ax = fig.add_subplot(111)
ax.grid(True, zorder=0)


# --- Data ---
ax.scatter(x, y, marker="_", s=100,
           label="Theoretical C(%) from PyCO2SYS", zorder=3)
ax.errorbar(x, y, yerr=y_error, fmt="none", capsize=3, zorder=3)

ax.scatter(
    fit_df["acid_total_mL"],
    fit_df["C_percent"],
    color="red", s=60, marker="x",
    label="Measured C(%) from exponential fits", zorder=5
)
ax.errorbar(fit_df["acid_total_mL"], fit_df["C_percent"],fit_df["Cerr"], color = "red", fmt= "none", capsize = 3)
# --- Shaded regions for fit windows ---
ax.axvspan(x_lin.iloc[0], x_lin.iloc[-1], 
           color='blue', alpha=0.12, zorder=1, label="Linear-fit region")

ax.axvspan(x.iloc[-10], x.iloc[-1], 
           color='green', alpha=0.12, zorder=1, label="Plateau region")

# --- Fitted piecewise line ---
x_fit = np.linspace(x.min(), x.max(), 300)
ax.plot(x_fit, linear_then_plateau(x_fit), 'r--',
        label="Linear→plateau model", zorder=4)

# # --- Labels ---
ax.set_xlabel("Titrant Volume (mL)")
ax.set_ylabel("C (%)")
ax.tick_params()
ax.legend()

fig.tight_layout()
plt.show()

#%%
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})
x = ttt.titrant_mass *1e3
x = ttt.pH
y = 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0]
y_error = 100 * co2s_fco2["u_dic"] / co2s_fco2["dic"][0]

# --- Linear fit using first 15 points ---
x_lin = x[:15]
y_lin = y[:15]
m, b = np.polyfit(x_lin, y_lin, 1)
print(f"Linear fit: slope = {m:.3f}, intercept = {b:.3f}")

# --- Plateau: average of last 10 points ---
plateau = np.mean(y[-10:])
print(f"Plateau value (avg of last 10 points) = {plateau:.3f}")

# --- Piecewise ---
def linear_then_plateau(x):
    x_intercept = (plateau - b) / m
    return np.where(x < x_intercept, m*x + b, plateau)

# --- Figure structure ---
fig = plt.figure(figsize=(7,5), dpi=250)
ax = fig.add_subplot(111)
ax.grid(True, zorder=0)


# --- Data ---
ax.scatter(x, y, marker="_", s=100,
           label="Theoretical C(%) from PyCO2SYS", zorder=3)
ax.errorbar(x, y, yerr=y_error, fmt="none", capsize=3, zorder=3)

ax.scatter(
    fit_df["acid_total_mL"],
    fit_df["C_percent"],
    color="red", s=60, marker="x",
    label="Measured C(%) from exponential fits", zorder=5
)
ax.errorbar(fit_df["acid_total_mL"], fit_df["C_percent"],fit_df["Cerr"], color = "red", fmt= "none", capsize = 3)
# --- Shaded regions for fit windows ---
ax.axvspan(x_lin.iloc[0], x_lin.iloc[-1], 
           color='blue', alpha=0.12, zorder=1, label="Linear-fit region")

ax.axvspan(x.iloc[-10], x.iloc[-1], 
           color='green', alpha=0.12, zorder=1, label="Plateau region")

# --- Fitted piecewise line ---
x_fit = np.linspace(x.min(), x.max(), 300)
ax.plot(x_fit, linear_then_plateau(x_fit), 'r--',
        label="Linear→plateau model", zorder=4)

# # --- Labels ---
ax.set_xlabel("Titrant Volume (mL)")
ax.set_ylabel("C (%)")
ax.tick_params()
ax.legend()

fig.tight_layout()
plt.show()