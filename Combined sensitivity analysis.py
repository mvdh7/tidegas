import numpy as np
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
# INPUT DATA  from sensitivity analysis slopes
# -------------------------------------------------------------
acids = np.array([
    0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35,
    1.5, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.55, 2.7, 2.85,
    3.0, 3.15, 3.3, 3.45, 3.6, 3.75, 3.9, 4.05, 4.2, 4.8
])

# Sensitivity slopes
slopes_C = np.array([
    -0.04806314, -0.0448846,  -0.04228354, -0.03942166, -0.03650324,
    -0.03365119, -0.03081317, -0.02785801, -0.02517493, -0.02191124,
    -0.01871132, -0.01558157, -0.01219426, -0.00922649, -0.00557336,
    -0.0033374,  -0.00035269, -0.00035046, -0.00036004, -0.00035178,
    -0.00034987, -0.00037529, -0.0003645,  -0.00036769, -0.00038471,
    -0.00037907, -0.00039564, -0.00038862, -0.00039807, -0.00042977
])

slopes_t = np.array([
    -0.00210319,  0.01416339,  0.02724774,  0.04168688,  0.05608967,
     0.07022771,  0.0838694,   0.09843407,  0.10990844,  0.12575498,
     0.14047181,  0.15535426,  0.17038026,  0.1816372,   0.19865413,
     0.20170006,  0.21728208,  0.21521675,  0.21095686,  0.21311262,
     0.21615972,  0.2018649,   0.20584499,  0.20350273,  0.19676874,
     0.19765147,  0.19115173,  0.19292562,  0.18927659,  0.17534257
])

slopes_k = np.array([
    -0.00285433,  0.01922175,  0.0369791,   0.05657508,  0.07612174,
     0.09530909,  0.11382282,  0.13358916,  0.14916154,  0.17066756,
     0.19064042,  0.21083804,  0.23123048,  0.24650776,  0.26960218,
     0.27373594,  0.29488299,  0.29208003,  0.28629875,  0.28922443,
     0.29335979,  0.27395966,  0.27936121,  0.27618242,  0.26704343,
     0.26824142,  0.25942035,  0.26182778,  0.25687551,  0.23796505
])

# -------------------------------------------------------------
# UNCERTAINTIES
# -------------------------------------------------------------
sigma_meas = 0.23604401444347173   # % DIC (constant offset)
sigma_k = 1                  # min^-1
sigma_t = 1.0                     # seconds
sigma_C = 1.0                     # %

# -------------------------------------------------------------
# ERROR CONTRIBUTIONS (% DIC)
# -------------------------------------------------------------
err_meas = np.full_like(acids, sigma_meas)

err_k = np.abs(slopes_k) * sigma_k
err_t = np.abs(slopes_t) * sigma_t
err_C = np.abs(slopes_C) * sigma_C

# Combined uncertainty (RSS)
err_total = np.sqrt(
    err_meas**2 +
    err_k**2 +
    err_t**2 +
    err_C**2
)

# -------------------------------------------------------------
# PLOT
# -------------------------------------------------------------
plt.figure()
plt.grid(alpha=0.4)

plt.scatter(acids, err_meas, linewidth=2,
         label="Measurement precision")

plt.scatter(acids, err_C, linewidth=2,
         label="C uncertainty (±1%)")

plt.scatter(acids, err_k, linewidth=2,
         label=r"$k$ uncertainty (±7%)")

plt.scatter(acids, err_t, linewidth=2,
         label="Time uncertainty (±5 s)")

plt.scatter(acids, err_total, color="black", linewidth=3,
         label="Total uncertainty (RSS)")

plt.xlabel("Titrant Volume (mL)")
plt.ylabel("DIC uncertainty (%)")

plt.legend(frameon=True)
plt.tight_layout()
plt.show()
#%%
import pandas as pd
df = pd.DataFrame({
    "titrant_volume_mL": acids[1:-1],
    "total_DIC_uncertainty_percent": err_total[1:-1]
})

# -------------------------------------------------------------
# SAVE TO CSV
# -------------------------------------------------------------
output_file = "DIC_total_uncertainty_vs_titrant_volume.csv"
df.to_csv(output_file, index=False)
