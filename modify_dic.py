# %%
import calkulate as calk
import PyCO2SYS as pyco2
from matplotlib import pyplot as plt

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

# Just take data from 30 October
dbs = dbs[(dbs.analysis_datetime.dt.month == 10) & (dbs.analysis_datetime.dt.day == 30)]
dbs["titrant_molinity"] = 0.09941
dbs["dic"] = 2220.91
dbs["temperature_override"] = 25.1  # as the files record 0 °C
dbs["salinity"] = 35.1  # TODO update
dbs["total_phosphate"] = 100
# dbs['total_silicate'] = 100
# TODO Workaround for density storing bug in v23.7.0
dbs["analyte_mass"] = (
    1e-3
    * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)

# Get one titration
tt = calk.to_Titration(dbs, 200)
ttt = tt.titration
tt.plot_alkalinity()
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

# # Plot expected DIC
# fig, ax = plt.subplots(dpi=300)
# ax.scatter(ttt.titrant_mass, co2s["dic"])
# ax.plot(ttt.titrant_mass, co2s["dic"] + co2s["u_dic"])
# ax.plot(ttt.titrant_mass, co2s["dic"] - co2s["u_dic"])
# ax.set_ylim(2000, 2400)

# # Plot alkalinity estimates through titration (tt.plot_alkalinity)
# fig, ax = plt.subplots(dpi=300)
# ax.scatter(ttt.titrant_mass, sr.alkalinity_all)
# ax.set_title(sr.alkalinity)

# # What should the C value be?
# co2s_fco2 = pyco2.sys(
#     par1=alkalinity_mixture.values,
#     par2=500,
#     par1_type=1,
#     par2_type=5,
#     temperature=ttt.temperature.values,
#     salinity=tt.salinity,
#     uncertainty_from={"par1": 5, "par2": 50},
#     uncertainty_into=["dic"],
# )

# fig, ax = plt.subplots(dpi=300)
# ax.scatter(ttt.titrant_mass * 1e3, 100 * co2s_fco2["dic"] / co2s_fco2["dic"][0])
