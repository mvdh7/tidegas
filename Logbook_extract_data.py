# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 10:26:52 2025

@author: nicor
"""

import re
import os
import pandas as pd
from datetime import datetime
import calkulate as calk  # your module
import shutil
import numpy as np

# Paths
excel_file = "logbook_automated_by_python_testing5.xlsx"
file_path = "data/vindta/r2co2/Nico"

# Read existing Excel logbook
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame()  # empty if file doesn't exist

# Import dbs file
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)
#%%
# create backups if not already done
tfiles = os.listdir(file_path)

for i, row in dbs.iterrows():
    datfile = os.path.join(row.file_path, row.file_name)
    
    # Backup if .bak doesn't exist
    if row.file_name[:-3] + "bak" not in tfiles:
        bakfile = datfile[:-3] + "bak"
        shutil.copyfile(datfile, bakfile)
# Regex: unified pattern to extract all info
file_pattern = re.compile(
    r'(junk|nose|CRM|sample|junk_old|nuts)-'         # sample type
    r'(\d{6})-'               # date YYMMDD
    r'(\d+)'                  # bottle number
    r'(?:-(\d+)_(\d+)mL)?'   # optional acid volume
    r'(?:-(\d+)_(\d+)incrmL)?',  # optional acid increment
    re.I
)
#%%
# Only .dat files
dat_files = [f for f in os.listdir(file_path) if f.lower().endswith('.dat')]

# Exclude aborted files manually
exceptions = [
    "0-0  0  (0)junk-250930-01.dat",
    "0-0  0  (0)junk-251001-06-2_25mL-0_15incrmL.dat",
    "0-0  0  (0)junk-251003-06-2_25mL-0_15incrmL.dat",
    "0-0  0  (0)junk-251014-06-4_2mL-0_15incrmL.dat",
    
]

valid_dat_files = [f for f in dat_files if f not in exceptions]

# Build rows
rows = []
emfvalstest = []
for i, file in enumerate(valid_dat_files):
    match = file_pattern.search(file.lower())
    if match:
        sample_type = match.group(1).capitalize()
        date_str = match.group(2)
        bottle = int(match.group(3))

        # Convert date to DD-MMM
        try:
            date_obj = datetime.strptime(date_str, "%y%m%d")
            date_fmt = date_obj.strftime("%d/%m/%Y")
        except:
            date_fmt = ""

        # Acid and increment (optional)
        acid_val = None
        incr_val = None
        emf_val = None
        if match.group(4) and match.group(5):
            acid_val = float(f"{match.group(4)}.{match.group(5)}")
        if match.group(6) and match.group(7):
            incr_val = float(f"{match.group(6)}.{match.group(7)}")
 
        # Read first EMF
        print(file)
        #have to setup if statement here to ensure it doesnt read files with only 1 line,
        #in those cases the titrant amount gives titrant_amount = data[:, col_titrant_amount]:
        # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        
        if incr_val is not None and incr_val <=2 and acid_val>0.15:
            dat_data = calk.read_dat(os.path.join(file_path, file))
        
            emf_val = (dat_data.measurement[0] if len(dat_data.measurement) > 0 else np.nan)
            emfvalstest.append(emf_val)
       
        rows.append({
            "date": date_fmt,
            "sample/junk": sample_type,
            "bottle": bottle,
            "file name alkalinity": file,
            "First emf value": emf_val,
            "acid added (mL)": acid_val,
            "acid increments (mL)": incr_val
        })

# Build new DataFrame
df_files = pd.DataFrame(rows)

# Sort by date then bottle
df_files['date_sort'] = pd.to_datetime(df_files['date'], format='%d/%m/%Y', errors='coerce')
df_files.sort_values(by=['date_sort', 'bottle'], inplace=True)
df_files.reset_index(drop=True, inplace=True)
df_files.drop(columns=['date_sort'], inplace=True)

# Columns we care about
cols_to_fill = [
    "date", "sample/junk", "bottle", "file name alkalinity",
    "First emf value", "acid added (mL)", "acid increments (mL)"
]

# Append only new files below existing Excel rows, preserving other columns
if not df.empty:
    existing_files = set(df["file name alkalinity"])
    new_rows = df_files[~df_files["file name alkalinity"].isin(existing_files)]
    
    # Create a DataFrame with all columns from the existing Excel
    new_rows_df = pd.DataFrame(columns=df.columns)
    
    # Populate only the columns we care about
    for col in cols_to_fill:
        if col in new_rows.columns:
            new_rows_df[col] = new_rows[col]
    
    # Append new rows
    df_updated = pd.concat([df, new_rows_df], ignore_index=True)
else:
    df_updated = df_files

# Save updated Excel
df_updated.to_excel(excel_file, index=False)
print(f"✅ Appended {len(new_rows) if not df.empty else len(df_updated)} new rows. Total rows now: {len(df_updated)}")
