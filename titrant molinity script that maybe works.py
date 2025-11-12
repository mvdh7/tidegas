# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:10:16 2025

@author: nicor
"""

import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np

file_path = "data/vindta/r2co2/Nico"
file = "data/vindta/r2co2/Nico/0-0  0  (0)CRM-251014-05-4_2mL-0_15incrmL.dat"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)


# dbs["cert. CRM CT"] = np.ones(221)*2029.34
# dbs["cert. CRM AT"] = np.ones(221)*2183.64
alkalinity_certified = np.empty(274)
alkalinity_certified[[98,267] ] = [2183.64, 2183.64]

dbs["salinity"] = 32.710
dbs["dic"] = 2029.34 
dbs["alkalinity_certified"] = alkalinity_certified
dbs.calibrate(98)
titrant_molinity_acid_batch_1 = dbs["titrant_molinity_here"][98]
print(f'titrant molinity = {dbs["titrant_molinity_here"][98]}')
# ds["alkalinity_certified"]=2183.64
# ds["dic"] = 2029.34
# ds["salinity"] = 32.710
dbs.calibrate(267)
titrant_molinity_acid_batch_2 = dbs["titrant_molinity_here"][267]
print(f'titrant molinity = {dbs["titrant_molinity_here"][267]}')
#%%
# # Now we can proceed with Calkulate as normal
# dbs["titrant_molinity"] = 0.1  # guess for now, calibrate later
# dbs["salinity"] = 30  # guess for now, measure later
# dbs["dic"] = 2350  # guess for now, measure later
# dbs["temperature_override"] = 25  # as the files record 0 °C


# dbs.solve()

# # TODO Workaround for density storing bug in v23.7.0
# dbs["analyte_mass"] = (
#     1e-3
#     * dbs.analyte_volume
#     / calk.density.seawater_1atm_MP81(
#         temperature=dbs.temperature_override, salinity=dbs.salinity
#     )
# )

# # Extract one titration to plot
# tt = dbs.to_Titration(98)
# tt.plot_pH()
# tt.plot_components()
# tt.plot_alkalinity()

# #%%
# print((dbs.alkalinity[98]))
# # TESTING CODE
# # automatically update calkulated alkalinity in logbook file

# reduced_dbs = dbs[["bottle","alkalinity","emf0","pH_init"]]
# test_alkalinity = np.array(dbs["alkalinity"])
# print(test_alkalinity)
# print(tt.file_name[:-4])