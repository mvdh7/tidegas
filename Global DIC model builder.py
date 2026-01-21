# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 12:49:12 2026

@author: nicor
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Build global DIC degassing model
#Polynomial fit of remaining DIC (%) vs titrant volume

# ------------------------------------------------------------------
# Load logbook
# ------------------------------------------------------------------
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

required_cols = [
    "Calculated in situ DIC (umol/kg)",
    "Reference DIC (umol/kg)",
    "acid added (mL)",
    "date",
    "waiting time (minutes)",
    "acid increments (mL)",
    "Mixing and waiting time (seconds)",
    "bottle"
]

if not all(c in df.columns for c in required_cols):
    raise ValueError("Missing required columns in logbook")
plot_df = df.dropna(subset=required_cols).copy()

# Experimental constraints
plot_df = plot_df[plot_df["waiting time (minutes)"] <= 0.05]
plot_df = plot_df[plot_df["acid increments (mL)"] <= 0.15]
plot_df = plot_df[plot_df["Mixing and waiting time (seconds)"] == 4]
plot_df = plot_df[plot_df["bottle"] != "1"]

# Date handling
plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

start_date = "2025-10-22"
plot_df = plot_df[plot_df["Date"] >= start_date]

# Known bad days
plot_df = plot_df[plot_df["date"]!= "10/11/2025"]
plot_df = plot_df[plot_df["date"]!= "7/11/2025"]
plot_df = plot_df[plot_df["date"]!= "5/11/2025"]

plot_df["Titrant Volume (ml)"] = pd.to_numeric(
    plot_df["acid added (mL)"], errors="coerce"
)

plot_df["In-situ DIC (%)"] = (
    100
    * plot_df["Calculated in situ DIC (umol/kg)"]
    / plot_df["Reference DIC (umol/kg)"]
)

grouped = plot_df.groupby("Titrant Volume (ml)")["In-situ DIC (%)"]

x = grouped.mean().index.values
y = grouped.mean().values
y_std = grouped.std().values

# Avoid singular weights
y_std = np.nan_to_num(y_std, nan=1.0)

fit_order = 4

coeffs, cov = np.polyfit(
    x,
    y,
    fit_order,
    cov=True
)

poly = np.poly1d(coeffs)
x_ref = np.linspace(0, 4.05, 29)
y_fit = poly(x_ref)

# Jacobian for uncertainty propagation
J = np.vstack([x_ref**i for i in range(fit_order, -1, -1)]).T
y_fit_err = np.sqrt(np.sum(J @ cov * J, axis=1))
plt.figure(figsize=(8,5))
plt.grid(True, alpha=0.4)

sns.scatterplot(
    data=plot_df,
    x="Titrant Volume (ml)",
    y="In-situ DIC (%)",
    alpha=0.4,
    label="Individual measurements"
)

plt.errorbar(
    x,
    y,
    yerr=y_std,
    fmt="o",
    color="red",
    capsize=4,
    label="Mean ± SD"
)

plt.plot(x_ref, y_fit, color="black", lw=2, label="Global degassing model")

plt.fill_between(
    x_ref,
    y_fit - y_fit_err,
    y_fit + y_fit_err,
    color="black",
    alpha=0.2,
    label="Model uncertainty"
)

plt.xlabel("Titrant volume (mL)")
plt.ylabel("Remaining DIC (%)")
plt.legend()
plt.tight_layout()
plt.show()

degassing_model = {
    "coeffs": coeffs,
    "cov": cov,
    "fit_order": fit_order,
    "x_ref": x_ref,
    "remaining_percent": y_fit,
    "remaining_percent_err": y_fit_err,
    "filters": {
        "waiting_time_max": 0.05,
        "acid_increment_max": 0.15,
        "mixing_time_s": 4
    },
}

np.save(
    "global_dic_degassing_model.npy",
    degassing_model,
    allow_pickle=True
)

print("Global DIC degassing model saved.")
