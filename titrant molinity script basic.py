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

dbs["salinity"] = 32.710
dbs["dic"] = 2029.34
alkalinity_certified = np.empty(338)*np.nan
alkalinity_certified[[267] ] = [2183.64]
dbs["alkalinity_certified"] = alkalinity_certified

dbs1 = dbs[dbs["bottle"]=="CRM-251014-05-4_2mL-0_15incrmL"]
dbs2 = dbs[dbs["bottle"]=="CRM-251110-06-4_2mL-0_15incrmL"]
#%%
#dbs.calkulate()
#%%
#dbs1.calibrate()
dbs.calibrate(267)
#%%
titrant_molinity_acid_batch2 = dbs["titrant_molinity_here"][267]
print(f'titrant molinity = {dbs["titrant_molinity_here"][267]}')
