# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 12:50:27 2026

@author: nicor
"""
import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np
file_path = "data/vindta/r2co2/Nico"
file = "data/vindta/r2co2/Nico/0-0  0  (0)CRM-251014-05-4_2mL-0_15incrmL.dat"
excel_file = "logbook_automated_by_python_testing.xlsx"


# -------------------------------------------------------------
# LOAD DBS AND EXCEL LOGBOOK
# -------------------------------------------------------------

log = pd.read_excel(excel_file).copy()
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

dbs["temperature_override"] = log["Temperature"]

#use temperature from the log, other parameters from DICKSON CRM #219
dbs["total_phosphate"] = 0.46
dbs["total_silicate"] = 7.4
dbs["total_nitrate"] = 4.1
dbs["total_nitrite"] = 0.01
dbs["salinity"] = 32.710
dbs["dic"] = 2029.34

alkalinity_certified = np.empty(338)*np.nan
alkalinity_certified[[98,267] ] = [2183.64, 2183.64]

dbs["alkalinity_certified"] = alkalinity_certified
#%%
dbs.calibrate(98)
titrant_molinity_acid_batch_1 = dbs["titrant_molinity_here"][98]
print(f'titrant molinity acid batch 1 = {dbs["titrant_molinity_here"][98]}')

dbs.calibrate(267)
titrant_molinity_acid_batch_2 = dbs["titrant_molinity_here"][267]
print(f'titrant molinity acid batch 2 = {dbs["titrant_molinity_here"][267]}')
