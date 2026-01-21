# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:35:32 2026

@author: nicor
"""

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
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 

#setup plotting parameters to make everything bigger
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 18,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 18,    # x tick labels
    "ytick.labelsize": 18,    # y tick labels
    "legend.fontsize": 16,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })


# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
excel_df = pd.read_excel(excel_file)


#set up day and month for filtering DBS and logbook file 
day = 30
month = 10
date = str(day)+"/"+str(month)+"/2025"


#%%
#update the dbs from the logbook, and specify a date
# TODO might as well make this smarter and use
dbs = dbs[(dbs.analysis_datetime.dt.month == month) & (dbs.analysis_datetime.dt.day == day)]
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]# -0.005 # Extract directly from excel, default = 0.1
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

# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure columns exist
if not all(col in df.columns for col in ["Calculated DIC (umol/kg)", "acid added (mL)", "date"]):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = df.dropna(subset=["Calculated DIC (umol/kg)", "acid added (mL)", "date", "waiting time (minutes)"]).copy()

#select only the files without any (significant) waiting time 
plot_df =plot_df[plot_df["waiting time (minutes)"]<=0.05]
plot_df =plot_df[plot_df["acid increments (mL)"]<=0.15]
plot_df = plot_df[plot_df["date"]=="30/10/2025"]

plot_df = plot_df[plot_df["bottle"]!="1"]

plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "30-10-2025"  # YYYY-MM-DD
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date


#plot_df = plot_df[plot_df["batch"]==1]

plot_df["Titrant Volume (ml)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Calculated in situ DIC (umol/kg)"] =  pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"], errors='coerce')
plot_df["DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC (%)"] = pd.to_numeric(100*plot_df["Calculated in situ DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC difference (umol)"] = pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"]-plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting Time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration Duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')

#%%

#load errors from seperate file 
dic_unc = pd.read_csv("DIC_total_uncertainty_vs_titrant_volume.csv")
plot_df = plot_df.merge(
    dic_unc,
    left_on="Titrant Volume (ml)",
    right_on="titrant_volume_mL",
    how="left"
)
#%%
# Scatter plot 
plt.figure()
plt.grid(True,alpha =0.4)
plt.scatter(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    s=50,
    label = "Measured DIC"
)

plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["DIC (%)"],yerr = 0.236,fmt="none", capsize=4, alpha=0.7)
plt.scatter(
    data=plot_df,color = 'red',
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    s=50,
    label = "In-situ DIC"
)
plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["In-situ DIC (%)"],color = 'red', yerr = plot_df["total_DIC_uncertainty_percent"],fmt="none", capsize=4, alpha=0.7)
plt.xlabel("Titrant Volume (mL)")
plt.ylabel("Remaining DIC (%)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#%%

import seaborn as sns
from scipy.optimize import curve_fit

# --- Exponential model ---
def exp_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Prepare data ---
fit_df = plot_df.sort_values(by="Titrant Volume (ml)")

# Include all points (no trimming)
x = fit_df["Titrant Volume (ml)"].values
y = fit_df["In-situ DIC (%)"].values
yerr = fit_df["total_DIC_uncertainty_percent"]
# Fit 4th-order polynomial: y = ax^4 + bx^3 + cx^2 + d+ e
fit_order = 4
coeffs, cov = np.polyfit(
    x[:-2],
    y[:-2],
    fit_order,
    cov=True
)
poly = np.poly1d(coeffs)
#%%
# 1-sigma uncertainties of coefficients
coeff_err = np.sqrt(np.diag(cov))

# Generate new x array from 0 → 4.05
x_new = np.linspace(0, 4.05, 28)
y_fit = poly(x_new)

# Build Jacobian
J = np.vstack([x_new**i for i in range(fit_order, -1, -1)]).T  # shape (len(x_new), 5)

# y-uncertainty from coefficient covariance
y_fit_err = np.sqrt(np.sum(J @ cov * J, axis=1))

plt.figure(figsize=(8,5))
plt.grid(True, alpha=0.4)

# Scatter original data
plt.errorbar(x, y, yerr=yerr, fmt='o',color= 'red',  capsize=4, alpha=0.7, label='In-situ DIC')

# Polynomial fit
plt.plot(x_new, y_fit, color='black', linewidth=2, label='Polynomial fit')

# Shaded error band
plt.fill_between(x_new, y_fit - y_fit_err, y_fit + y_fit_err,
                 color='black', alpha=0.2, label='Fit uncertainty')

plt.xlabel("Titrant Volume (mL)")
plt.ylabel("Remaining DIC (%)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
#%%

param_names = ["a", "b", "c", "d", "e"]

text_lines = [
    f"{name} = {val:.1e} ± {err:.1e}"
    for name, val, err in zip(param_names, coeffs, coeff_err)
]

fit_text = "4th-order polynomial fit\n" + "\n".join(text_lines)
# --- Convert % DIC to absolute DIC ---
ref_dic = plot_df["Reference DIC (umol/kg)"].iloc[0]
absolute_dic = (y_fit / 100) * ref_dic

# --- Store in dataframe ---
fit_df_out = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": absolute_dic
})


fit_df.head()

# Scatter plot 
plt.figure()
plt.grid(True,alpha =0.4)
plt.scatter(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    s=50,
    label = "Measured DIC",alpha = 0.7
)

plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["DIC (%)"],yerr = 0.236,fmt="none", capsize=4, alpha=0.7)
plt.scatter(
    data=plot_df,color = 'red',
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    s=50,
    label = "In-situ DIC",alpha = 0.7
)
plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["In-situ DIC (%)"],color = 'red', yerr = plot_df["total_DIC_uncertainty_percent"],fmt="none", capsize=4, alpha=0.7)
plt.xlabel("Titrant Volume (mL)")
plt.ylabel("Remaining DIC (%)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Plot fitted exponential model
plt.plot(
    fit_df_out["Titrant Volume (ml)"],
    fit_df_out["Fitted In-situ DIC (%)"],
    label="Polynomial fit",
    linewidth=2, color = 'black',zorder=-1
)
# Text box with fit parameters
plt.text(
    0.02, 0.98,
    fit_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

#%% 
#combine the fit with the alkalinity script

#get one titration
tt = calk.to_Titration(dbs, 200)
ttt = tt.titration
#tt.plot_alkalinity()
totals = {k: ttt[k].values for k in ttt.columns if k.startswith("total_") or k == "dic"}
ttt["Titrant Volume (ml)"] = np.linspace(0, 4.05,len(ttt.titrant_mass.values))

totals["dic"] = np.array((fit_df_out["Absolute DIC (umol/kg)"]+ 127.1162658006374)*1e-6)
#totals["dic"] = np.array((fit_df_out["Absolute DIC (umol/kg)"])*1e-6)
totals["dic"] = np.array(ttt["dic"])
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

offset = co2s["dic"][0]-fit_df_out["Absolute DIC (umol/kg)"][0]
print(f'offset is {offset} umol/kg')
#%% Plot expected DIC with shaded uncertainty
plt.figure(figsize=(8,5))

# Shaded region for theoretical DIC uncertainty
plt.fill_between(
    ttt["Titrant Volume (ml)"],
    co2s["dic"] - co2s["u_dic"],
    co2s["dic"] + co2s["u_dic"],
    color='lightblue',
    alpha=0.3,
    label="Theoretical uncertainty"
)

# Theoretical DIC line
plt.scatter(ttt["Titrant Volume (ml)"], co2s["dic"], color='blue', label="Theoretical DIC prediction")

# Model points
plt.scatter(ttt["Titrant Volume (ml)"],
            ttt["dic"]*1e6,
            color='purple',
            label="DIC dilution")

# Model points
plt.scatter(fit_df_out["Titrant Volume (ml)"],
            fit_df_out["Absolute DIC (umol/kg)"],
            color='green',
            label="DIC degassing")

# Model shifted points
plt.scatter(fit_df_out["Titrant Volume (ml)"] + 0,  # if offset is already applied separately
            fit_df_out["Absolute DIC (umol/kg)"] + offset,
            color='orange',
            label="DIC degassing shifted")

plt.xlabel("Titrant Volume (mL)")
plt.ylabel("DIC (µmol/kg)")
plt.legend()
plt.ylim(1500, 2800)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
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
dic_variants = {"DIC dilution":     np.array(ttt["dic"]),
   "DIC degassing":    np.array((fit_df_out["Absolute DIC (umol/kg)"]) * 1e-6),
   "DIC degassing shfited":  np.array((fit_df_out["Absolute DIC (umol/kg)"] + offset) * 1e-6),
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
        "alkalinity_error":  np.std(sr_var.alkalinity_all[ttt.used]), 
        "variant": name
    })
  
    variant_results[name] = df_var

#%%
# ---------------------------------------------------------------
# Plot alkalinity for all DIC variants
# ---------------------------------------------------------------
plt.figure(figsize=(9, 6))
# ---------------------------------------------------------------
# Build table data
# ---------------------------------------------------------------
table_rows = []
for name, dfv in variant_results.items():
    final_val = dfv["alkalinity_final"].iloc[0]
    std_val   = dfv["alkalinity_error"].iloc[0]
    table_rows.append([
        name,
        f"{final_val:.1f} ± {std_val:.1f}"
    ])

# Column labels
col_labels = ["Variant", "Total Alkalinity (µmol/kg)"]

# ---------------------------------------------------------------
# Add table to plot
# ---------------------------------------------------------------
table = plt.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="upper right",
    colLoc="center",
    cellLoc="center",
    bbox=[0.62, 0.05, 0.36, 0.35]  # [left, bottom, width, height]
)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.2)

# Slight transparency so plot remains visible
for _, cell in table.get_celld().items():
    cell.set_alpha(0.85)
    
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
shade_end   = dfv["Titrant Volume (ml)"].iloc[23]

ax.axvspan(shade_start, shade_end, color="gray", alpha=0.25, label="Used region")
ax.axhline(tt.alkalinity,alpha =0.5, color = 'black',  label="Final value", zorder=-1)
plt.xlabel("Titrant volume (mL)")
plt.ylabel("Alkalinity (µmol/kg)")
#plt.title("Alkalinity vs. Titrant Mass for Different DIC Variants")
plt.legend(loc= 'lower center')
plt.tight_layout()
plt.show()


#%% Prepare variant results (same as before)
dic_variants = {
    "DIC dilution": np.array(ttt["dic"]),
    "DIC degassing": np.array(fit_df_out["Absolute DIC (umol/kg)"] * 1e-6),
    "DIC degassing shifted": np.array((fit_df_out["Absolute DIC (umol/kg)"] + offset) * 1e-6),
    "DIC degassing shifted_upper": np.array((fit_df_out["Absolute DIC (umol/kg)"]*(1+y_fit_err/100) + offset) * 1e-6),
    "DIC degassing shifted_lower": np.array((fit_df_out["Absolute DIC (umol/kg)"]*(1-y_fit_err/100) + offset) * 1e-6),
    "DIC degassing_upper": np.array(fit_df_out["Absolute DIC (umol/kg)"]*(1+y_fit_err/100) * 1e-6),
    "DIC degassing_lower": np.array(fit_df_out["Absolute DIC (umol/kg)"]*(1-y_fit_err/100) * 1e-6)
}

variant_results = {}

for name, dic_array in dic_variants.items():
    totals_var = base_totals.copy()
    totals_var["dic"] = dic_array

    # Solve alkalinity
    sr_var = calk.core.solve_emf(
        tt.titrant_molinity,
        ttt.titrant_mass.values,
        ttt.emf.values,
        ttt.temperature.values,
        tt.analyte_mass,
        totals_var,
        k_constants,
    )

    # Store results
    df_var = pd.DataFrame({
        "titrant_mass": ttt.titrant_mass.values,
        "Titrant Volume (ml)": np.linspace(0, 4.05, len(ttt.titrant_mass)),
        "alkalinity_all": sr_var.alkalinity_all,
        "alkalinity_final": sr_var.alkalinity,
        "alkalinity_error":  np.std(sr_var.alkalinity_all[ttt.used]), 
        "variant": name
    })

    variant_results[name] = df_var


#%% Plot alkalinity with DIC-propagated error bands and table

plt.figure(figsize=(9,6))
plt.grid(True, alpha=0.4)
plt.ylim(2300, 2485)
ax = plt.gca()

# Define colors for clarity
colors = {
    "DIC dilution": "blue",
    "DIC degassing": "green",
    "DIC degassing shifted": "orange"
}

# Plot DIC standard
df_standard = variant_results["DIC dilution"]
plt.plot(
    df_standard["Titrant Volume (ml)"],
    df_standard["alkalinity_all"],
    color=colors["DIC dilution"],
    marker='o',
    label="DIC dilution"
)

# Plot DIC fit and DIC fit offset with shaded propagated error
for base_name in ["DIC degassing", "DIC degassing shifted"]:
    dfv = variant_results[base_name]
    plt.plot(
        dfv["Titrant Volume (ml)"],
        dfv["alkalinity_all"],
        color=colors[base_name],
        marker='s',
        label=base_name
    )

    # Shaded uncertainty from DIC upper/lower
    upper_name = f"{base_name}_upper"
    lower_name = f"{base_name}_lower"
    df_upper = variant_results[upper_name]
    df_lower = variant_results[lower_name]

    ax.fill_between(
        dfv["Titrant Volume (ml)"],
        df_lower["alkalinity_all"],
        df_upper["alkalinity_all"],
        color=colors[base_name],
        alpha=0.2
    )

# Highlight "used" titrant region
shade_start = dfv["Titrant Volume (ml)"].iloc[17]
shade_end   = dfv["Titrant Volume (ml)"].iloc[23]
ax.axvspan(shade_start, shade_end, color="gray", alpha=0.25, label="Used region")

# Final alkalinity reference line
ax.axhline(tt.alkalinity, alpha=0.5, color='black', label="Final value", zorder=-1)

# Axis labels
plt.xlabel("Titrant volume (mL)")
plt.ylabel("Alkalinity (µmol/kg)")

# --- Build table with mean ± error from final alkalinity ---
table_rows = []
for name in ["DIC dilution", "DIC degassing", "DIC degassing shifted"]:
    df_base = variant_results[name]

    # Intrinsic std in used region
    std_used = np.std(df_base["alkalinity_all"][ttt.used])

    # Propagated DIC error if available
    upper_name = f"{name}_upper"
    lower_name = f"{name}_lower"
    if upper_name in variant_results:
        df_upper = variant_results[upper_name]
        df_lower = variant_results[lower_name]
        prop_error = (df_upper["alkalinity_all"][ttt.used] - df_lower["alkalinity_all"][ttt.used]) / 2
        prop_error_mean = np.mean(prop_error)
    else:
        prop_error_mean = 0.0
        
    print(f'error in alkalinity from std dev = {std_used} , from propagating DIC = {prop_error_mean}')
    # Combine in quadrature
    total_error = np.sqrt(std_used**2 + prop_error_mean**2)

    # Mean alkalinity in used region
    alk_final = np.mean(df_base["alkalinity_all"][ttt.used])
    
    table_rows.append([name, f"{alk_final:.1f} ± {total_error:.1f}"])

col_labels = ["Variant", "Total Alkalinity (µmol/kg)"]
table = plt.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="upper right",
    colLoc="center",
    cellLoc="center",
    bbox=[0.64, 0.4, 0.36, 0.35]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.2)
for _, cell in table.get_celld().items():
    cell.set_alpha(0.85)

plt.legend(loc= "lower right")
plt.tight_layout()
plt.show()
