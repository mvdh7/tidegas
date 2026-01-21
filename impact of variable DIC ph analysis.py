# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 11:42:04 2026

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
# Scatter plot with hue by date
plt.figure()
plt.grid(True,alpha =0.4)
sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    palette="tab20",
    s=70,
    label = "Measured DIC"
)

plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["DIC (%)"],yerr = 1,fmt="none", capsize=4, alpha=0.7)
sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    palette="tab20",
    s=70,
    label = "In-situ DIC"
)
plt.errorbar(x=plot_df["Titrant Volume (ml)"],y=plot_df["In-situ DIC (%)"],yerr = 1,fmt="none", capsize=4, alpha=0.7)
plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")
plt.title("DIC vs Acid Added (11/11)")
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

# Fit 4th-order polynomial: y = ax^4 + bx^3 + cx^2 + d+ e
coeffs = np.polyfit(x[:-2], y[:-2], 4)   # returns [a, b, c, d, e]
poly = np.poly1d(coeffs)       # convenient polynomial object

# Generate new x array from 0 → 4.05
x_new = np.linspace(0, 4.05, 28)
y_fit = poly(x_new)


# --- Convert % DIC to absolute DIC ---
ref_dic = plot_df["Reference DIC (umol/kg)"].iloc[0]
absolute_dic = (y_fit / 100) * ref_dic

# --- Store in dataframe ---
fit_df_out = pd.DataFrame({
    "Titrant Volume (ml)": x_new,
    "Fitted In-situ DIC (%)": y_fit,
    "Absolute DIC (umol/kg)": absolute_dic
})

#%%


fit_df.head()
plt.figure()
plt.grid(True, alpha=0.4)

sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="DIC (%)",
    s=70,
    label="Measured DIC"
)

plt.errorbar(
    x=plot_df["Titrant Volume (ml)"],
    y=plot_df["DIC (%)"],
    yerr=1,
    fmt="none",
    capsize=4,
    alpha=0.7
)

sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    s=70,
    label="In-situ DIC"
)

plt.errorbar(
    x=plot_df["Titrant Volume (ml)"],
    y=plot_df["In-situ DIC (%)"],
    yerr=1,
    fmt="none",
    color= "orange",
    capsize=4,
    alpha=0.7
)

# Plot fitted exponential model
plt.plot(
    fit_df_out["Titrant Volume (ml)"],
    fit_df_out["Fitted In-situ DIC (%)"],
    label="Exponential fit",
    linewidth=2
)

plt.xlabel("Titrant Volume (ml)")
plt.ylabel("Remaining DIC (%)")
plt.title("DIC vs Acid Added (11/11)")
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

#totals["dic"] = np.array((fit_df_out["Absolute DIC (umol/kg)"]+127.46661137468027)*1e-6)
#totals["dic"] = np.array((fit_df_out["Absolute DIC (umol/kg)"])*1e-6)
totals["dic"] = np.ones(len(ttt.titrant_mass.values))*(ref_dic)*1e-6
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
#%%
# Plot expected DIC
plt.figure()
plt.scatter(ttt["Titrant Volume (ml)"], co2s["dic"],label = "Theoretical DIC prediction")
plt.plot(ttt["Titrant Volume (ml)"], co2s["dic"] + co2s["u_dic"])
plt.plot(ttt["Titrant Volume (ml)"], co2s["dic"] - co2s["u_dic"])
plt.scatter(fit_df_out["Titrant Volume (ml)"],fit_df_out["Absolute DIC (umol/kg)"], label = "Model")
plt.scatter(fit_df_out["Titrant Volume (ml)"],fit_df_out["Absolute DIC (umol/kg)"]+offset, label = "Model shifted")
plt.xlabel("Titrant Volume (ml)")
plt.ylabel("DIC (umol/kg)")
plt.legend()
plt.ylim(1500, 2800)

#%%
# # plot Expected DIC with pH 
# plt.figure()
# plt.scatter(ttt.pH.values, fit_df_out["Absolute DIC (umol/kg)"][0]-fit_df_out["Absolute DIC (umol/kg)"],label = "Modelled DIC loss")
# plt.ylabel("DIC (umol/kg)")
# plt.xlabel("pH")
# plt.gca().invert_xaxis()
#%%
# plot calculated HCO3 and CO2 with pH 
plt.figure()
plt.grid(alpha = 0.4)
plt.scatter(ttt.pH.values, ttt.CO3.values*1e6, label = "CO3")
plt.scatter(ttt.pH.values, ttt.HCO3.values*1e6,label = "HCO3")
plt.scatter(ttt.pH.values, ttt.dic.values*1e6-ttt.HCO3.values*1e6-ttt.CO3.values*1e6 , label = "CO2")
plt.scatter(ttt.pH.values, ttt.dic.values*1e6 , label = "DIC")
plt.scatter(ttt.pH.values, ttt.H.values*1e6,label = "H")
plt.scatter(ttt.pH.values, fit_df_out["Absolute DIC (umol/kg)"], label = "modelled DIC",marker= "x")
plt.ylabel("Concentration (umol/kg)")
plt.xlabel("pH")
plt.gca().invert_xaxis()
plt.legend()

Concentration_H_added = 8*0.15*(1/tt.analyte_mass)*tt.titrant_molinity/1000*1e6 #how much umol acid is added
#%%
# plot calculated HCO3 and CO2 with pH 
plt.figure()
plt.grid(alpha = 0.4)
# plt.scatter(ttt.pH.values, ttt.CO3.values*1e6, label = "CO3")
# plt.scatter(ttt.pH.values, ttt.HCO3.values*1e6,label = "HCO3")
# plt.scatter(ttt.pH.values, ttt.dic.values*1e6-ttt.HCO3.values*1e6-ttt.CO3.values*1e6 , label = "CO2")
# plt.scatter(ttt.pH.values, ttt.dic.values*1e6 , label = "DIC")
plt.scatter(ttt.pH.values, ttt.H.values*1e6 - (ttt.dic.values*1e6 -fit_df_out["Absolute DIC (umol/kg)"])*(ttt.HCO3.values/(ttt.HCO3.values+(ttt.dic.values-ttt.HCO3.values-ttt.CO3.values))),label = "H-adjusted")
plt.scatter(ttt.pH.values, ttt.H.values*1e6,label = "H")
#plt.scatter(ttt.pH.values, fit_df_out["Absolute DIC (umol/kg)"], label = "modelled DIC",marker= "x")
plt.ylabel("Concentration (umol/kg)")
plt.xlabel("pH")
plt.gca().invert_xaxis()
plt.legend()

Concentration_H_added = 8*0.15*(1/tt.analyte_mass)*tt.titrant_molinity/1000*1e6 #how much umol acid is added
#%%
H_added = np.linspace(0,len(ttt.H.values)-1,len(ttt.H.values)) *0.15*(1/tt.analyte_mass)*tt.titrant_molinity/1000*1e6
# plot calculated HCO3 and CO2 with pH 
plt.figure()
plt.grid(alpha = 0.4)
# plt.scatter(ttt.pH.values, ttt.CO3.values*1e6, label = "CO3")
# plt.scatter(ttt.pH.values, ttt.HCO3.values*1e6,label = "HCO3")
# plt.scatter(ttt.pH.values, ttt.dic.values*1e6-ttt.HCO3.values*1e6-ttt.CO3.values*1e6 , label = "CO2")
# plt.scatter(ttt.pH.values, ttt.dic.values*1e6 , label = "DIC")
plt.scatter(ttt.pH.values, (H_added - (ttt.dic.values*1e6 -fit_df_out["Absolute DIC (umol/kg)"])*(ttt.HCO3.values/(ttt.HCO3.values+(ttt.dic.values-ttt.HCO3.values-ttt.CO3.values))))/H_added,label = r"$\frac{\text{Effective Titrant molinity}}{\text{Titrant molinity}}$")
#plt.scatter(ttt.pH.values, H_added,label = "H")
#plt.scatter(ttt.pH.values, fit_df_out["Absolute DIC (umol/kg)"], label = "modelled DIC",marker= "x")
plt.ylabel("Molinity ratio")
plt.xlabel("pH")
plt.gca().invert_xaxis()
plt.legend(loc = "lower right")

#%%
H_added = np.linspace(0,len(ttt.H.values)-1,len(ttt.H.values)) *0.15*(1/tt.analyte_mass)*tt.titrant_molinity/1000*1e6
# plot calculated HCO3 and CO2 with pH 
plt.figure()
plt.grid(alpha = 0.4)
plt.scatter(ttt.pH.values, ttt.CO3.values/ttt.dic.values, label = r"$CO^{2-}_3$")
plt.scatter(ttt.pH.values, ttt.HCO3.values/ttt.dic.values,label = r"$HCO^-_3$")
plt.scatter(ttt.pH.values, (ttt.dic.values*1e6-ttt.HCO3.values*1e6-ttt.CO3.values*1e6)/(ttt.dic.values*1e6) , label = r"$CO_2$")
#plt.scatter(ttt.pH.values, ttt.dic.values/ttt.dic.values, label = "DIC")
#plt.scatter(ttt.pH.values, (H_added - (ttt.dic.values*1e6 -fit_df_out["Absolute DIC (umol/kg)"])*(ttt.HCO3.values/(ttt.HCO3.values+(ttt.dic.values-ttt.HCO3.values-ttt.CO3.values))))/H_added,label = r"$\frac{\text{Effective Titrant molinity}}{\text{Titrant molinity}}$")
#plt.scatter(ttt.pH.values, H_added,label = "H")
#plt.scatter(ttt.pH.values, fit_df_out["Absolute DIC (umol/kg)"], label = "modelled DIC",marker= "x")
plt.ylabel("Ratios of concentration")
plt.xlabel("pH")
plt.gca().invert_xaxis()
plt.legend(loc = "right")

#%%
plt.figure()

# --- Shaded area for theoretical DIC ± uncertainty ---
plt.fill_between(
    ttt["Titrant Volume (ml)"],
    co2s["dic"] - co2s["u_dic"],
    co2s["dic"] + co2s["u_dic"],
    color="teal",
    alpha=0.4,
    label="Theoretical DIC ± uncertainty"
)

# --- Plot theoretical DIC line ---
plt.scatter(
    ttt["Titrant Volume (ml)"],
    co2s["dic"],
    color="blue",
    s = 80,
    marker = "X",
    label="Theoretical DIC"
)

# --- Model points ---
plt.scatter(
    fit_df_out["Titrant Volume (ml)"],
    fit_df_out["Absolute DIC (umol/kg)"],
    color="orange",
    s=60,
    label="Model"
)

plt.scatter(
    fit_df_out["Titrant Volume (ml)"],
    fit_df_out["Absolute DIC (umol/kg)"] + offset,
    color="red",
    s=60,
    label="Model shifted"
)

plt.xlabel("Titrant Volume (ml)")
plt.ylabel("DIC (µmol/kg)")
plt.ylim(1500, 2800)
plt.title("DIC vs. Titrant Volume")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Plot alkalinity estimates through titration (tt.plot_alkalinity)
fig, ax = plt.subplots(dpi=300)
ax.scatter(ttt.titrant_mass, sr.alkalinity_all)
ax.set_title(f'$T_A=${sr.alkalinity:.3f} $\pm$ {sr.alkalinity_std:.3f} $\mu mol/kg$' )
ax.set_ylim(2350,2550)
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

# fig, ax = plt.subplots(dpi=300)
# ax.scatter(ttt.titrant_mass * 1e3, 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0])