# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 12:52:59 2026

@author: nicor
"""
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calkulate as calk
import PyCO2SYS as pyco2



#Apply global DIC degassing model to all reference titrations
model = np.load(    "global_dic_degassing_model.npy",allow_pickle=True).item()

coeffs = model["coeffs"]

def dic_fraction(x):
    """Fraction of initial DIC remaining"""
    return np.poly1d(coeffs)(x) / 100.0


# Paths
dbs_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=dbs_path)

excel_file = "logbook_automated_by_python_testing.xlsx"
excel_df = pd.read_excel(excel_file)
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


excel_df["Date"] = pd.to_datetime(excel_df["date"], format="%d/%m/%Y")

ref_df = excel_df[
    (excel_df["Alkalinity daily reference measurement"] == 1) &
    (excel_df["waiting time (minutes)"] <= 0.05) &
    (excel_df["acid increments (mL)"] <= 0.15) &
    (excel_df["Mixing and waiting time (seconds)"] == 4)]


#Exclude nuts measurements, as these have not been used at all in the DIC model (and have different alkalinity and DIC)
ref_df = ref_df[ref_df["sample/junk"]!= "Nuts"]

# # Filter from a certain date onwards
start_date = "10-22-2025"  # YYYY-MM-DD
ref_df = ref_df[ref_df["Date"] >= start_date]

ref_df = ref_df[ref_df["Date"]!= "11-10-2025"]
ref_df = ref_df[ref_df["Date"]!= "11-7-2025"]
ref_df = ref_df[ref_df["Date"]!= "11-5-2025"]

titration_indices = ref_df["Titration index"].unique()

print(f"Found {len(titration_indices)} reference titrations")

profiles_all = []
summary_rows = []

for ti in titration_indices:

    print(f"Processing titration {ti}")

    # ---------------------------
    # Load titration
    # ---------------------------
    tt = calk.to_Titration(dbs, ti)
    ttt = tt.titration

    x = np.linspace(0, 4.05, len(ttt.titrant_mass))

    # ---------------------------
    # Reference DIC and sample type
    # ---------------------------
    ref_dic = ref_df.loc[
        ref_df["Titration index"] == ti,
        "Reference DIC (umol/kg)"
    ].iloc[0]
    
    ref_sample = ref_df.loc[
        ref_df["Titration index"] == ti,
        "sample/junk"
    ].iloc[0]
    # ---------------------------
    # Build degassing DIC curve
    # ---------------------------
    dic_deg = dic_fraction(x) * ref_dic * 1e-6  # mol/kg

    # ---------------------------
    # Base totals
    # ---------------------------
    base_totals = {
        k: ttt[k].values
        for k in ttt.columns
        if k.startswith("total_") or k == "dic"
    }

    k_constants = {
        k: ttt[k].values
        for k in [
            "k_alpha", "k_ammonia", "k_beta", "k_bisulfate", "k_borate",
            "k_carbonic_1", "k_carbonic_2", "k_fluoride",
            "k_phosphoric_1", "k_phosphoric_2", "k_phosphoric_3",
            "k_silicate", "k_sulfide", "k_water",
        ]
    }

    # ---------------------------
    # Offset calculation
    # ---------------------------
    totals_base = base_totals.copy()
    totals_base["dic"] = np.array(ttt["dic"])

    sr_base = calk.core.solve_emf(
        tt.titrant_molinity,
        ttt.titrant_mass.values,
        ttt.emf.values,
        ttt.temperature.values,
        tt.analyte_mass,
        totals_base,
        k_constants,
    )

    alkalinity_mix = (
        sr_base.alkalinity * tt.analyte_mass
        - 1e6 * tt.titrant_molinity * ttt.titrant_mass
    ) / (tt.analyte_mass + ttt.titrant_mass)

    co2s = pyco2.sys(
        par1=alkalinity_mix.values,
        par2=ttt.pH.values,
        par1_type=1,
        par2_type=3,
        temperature=ttt.temperature.values,
        salinity=tt.salinity,
        opt_pH_scale=3,
    )

    offset = co2s["dic"][0] - ref_dic*dic_fraction(0)

    # ---------------------------
    # Apply degassing + offset
    # ---------------------------
    totals_deg = base_totals.copy()
    totals_deg["dic"] = (dic_deg * 1e6 + offset) * 1e-6

    sr_deg = calk.core.solve_emf(
        tt.titrant_molinity,
        ttt.titrant_mass.values,
        ttt.emf.values,
        ttt.temperature.values,
        tt.analyte_mass,
        totals_deg,
        k_constants,
    )

    # ---------------------------
    # Store profile
    # ---------------------------
    df_prof = pd.DataFrame({
        "Titration index": ti,
        "Titrant Volume (ml)": x,
        "Alkalinity (umol/kg)": sr_deg.alkalinity_all,
        "Offset (umol/kg)": offset,
        "Sample": ref_sample
    })

    profiles_all.append(df_prof)

    # ---------------------------
    # Summary stats
    # ---------------------------
    alk_used = sr_deg.alkalinity_all[ttt.used]

    summary_rows.append({
        "Titration index": ti,
        "Reference DIC (umol/kg)": ref_dic,
        "Offset (umol/kg)": offset,
        "Alkalinity mean (umol/kg)": np.mean(alk_used),
        "Alkalinity std (umol/kg)": np.std(alk_used),
    })
#%%

profiles_df = pd.concat(profiles_all, ignore_index=True)
summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter("degassing_corrected_reference_titrations.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    profiles_df.to_excel(writer, sheet_name="Alkalinity profiles", index=False)

print("Excel output written.")

#%%
plt.figure(figsize=(9,6))
plt.grid(True, alpha=0.3)

for ti in titration_indices:
    dfp = profiles_df[profiles_df["Titration index"] == ti]
    dfs = summary_df[summary_df["Titration index"] == ti]
    alk = dfp["Alkalinity (umol/kg)"].values
    alk_norm = alk - dfs["Alkalinity mean (umol/kg)"].values  # normalize near equivalence
    
    
    plt.plot(
        dfp["Titrant Volume (ml)"],
        alk_norm,
        alpha=0.4
    )

plt.xlabel("Titrant volume (mL)")
plt.ylabel("Δ Alkalinity (µmol/kg)")
plt.title("Normalized alkalinity curves (degassing corrected)")
plt.tight_layout()
plt.show()



#%%
samples_to_plot = ["Nose", "Junk_old", "Junk_old_2", "Junk_old_3"]
# Unique samples actually present
unique_samples = profiles_df["Sample"].unique()

# Use a qualitative colormap
cmap = plt.get_cmap("tab10")

sample_colors = {
    sample: cmap(i % cmap.N)
    for i, sample in enumerate(unique_samples)
}
# Use a qualitative colormap
unique_samples = profiles_df["Sample"].unique()

cmap = plt.get_cmap("plasma")

# Evenly spaced colors across plasma
sample_colors = {
    sample: cmap(i / max(len(unique_samples) - 1, 1))
    for i, sample in enumerate(unique_samples)
}

# for sample in samples_to_plot:

#     plt.figure(figsize=(9,6))
#     plt.grid(True, alpha=0.3)

#     df_sample = profiles_df[profiles_df["Sample"] == sample]

#     if df_sample.empty:
#         print(f"No data for sample: {sample}")
#         continue

#     for ti in df_sample["Titration index"].unique():

#         dfp = df_sample[df_sample["Titration index"] == ti]
#         dfs = summary_df[summary_df["Titration index"] == ti]

#         alk = dfp["Alkalinity (umol/kg)"].values
#         alk_norm = alk - dfs["Alkalinity mean (umol/kg)"].values
        
#         if max(alk_norm) >= 10:
#             print(ti)
#         plt.plot(
#             dfp["Titrant Volume (ml)"],
#             alk_norm,
#             color=sample_colors[sample],
#             alpha=0.5
#         )

#     plt.xlabel("Titrant volume (mL)")
#     plt.ylabel("Δ Alkalinity (µmol/kg)")
#     plt.title(f"Normalized alkalinity curves – {sample}")
#     plt.tight_layout()
#     plt.show()


#%%
plt.figure(figsize=(10,6))
plt.grid(True, alpha=0.3)

for sample in samples_to_plot:

    df_sample = profiles_df[profiles_df["Sample"] == sample]

    for ti in df_sample["Titration index"].unique():

        dfp = df_sample[df_sample["Titration index"] == ti]
        dfs = summary_df[summary_df["Titration index"] == ti]

        alk = dfp["Alkalinity (umol/kg)"].values
        alk_norm = alk - dfs["Alkalinity mean (umol/kg)"].values

        plt.plot(
            dfp["Titrant Volume (ml)"],
            alk_norm,
            color=sample_colors[sample],
            alpha=0.4,
            label=sample if ti == df_sample["Titration index"].unique()[0] else ""
        )

plt.xlabel("Titrant volume (mL)")
plt.ylabel("Δ Alkalinity (µmol/kg)")
plt.legend()
plt.tight_layout()
plt.show()
