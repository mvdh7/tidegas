import re
import os
import pandas as pd
from datetime import datetime
import calkulate as calk
import shutil
import numpy as np
# Path to your existing logbook excel file
excel_file = "logbook_automated_by_python_testing.xlsx"

# Folder containing your data files
file_path = "data/vindta/r2co2/Nico"
# Import dbs file
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

# For each titration file, if there isn't already a backup copy:
# 1. Make a backup copy (with extension .bak).
# 2. Extract the value of the first EMF reading and write it in the log
# 3. Extract the file name, type of sample (Junk/other/etc.), date and number from the file name and write it in the log


tfiles = os.listdir(file_path)
#create an empty list for storing ALL first EMF values, which can be added to the dataframe later
EMF_first = []

for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    if row.file_name[:-3] + "bak" not in tfiles:
        print(row.file_name)
        bakfile = datfile[:-3] + "bak" #has to be a better way to do this
        shutil.copyfile(datfile, bakfile) #copy the datafile into a backup file (still without correct acid addition)
    
    dat_data = calk.read_dat(datfile)
    EMF_first.append(dat_data.measurement[0]) #append the first measurement of EMF 
    
  
        
        
# Read existing Excel
df = pd.read_excel(excel_file)

# Pattern: match junk/sample, YYMMDD, bottle number
pattern = re.compile(r'(junk|sample)-(\d{6})-(\d+)', re.I)


# Only process .dat files
dat_files = [f for f in os.listdir(file_path) if f.lower().endswith('.dat')]

#create exceptions for single line dat files (aborted measurements)
dat_files.remove("0-0  0  (0)junk-250930-01.dat")


rows = []

for file in dat_files:
    match = pattern.search(file.lower())
    if match:
        sample_type = match.group(1).capitalize()
        date_str = match.group(2)
        bottle = int(match.group(3))

        # Convert date to DD-MMM
        try:
            date_obj = datetime.strptime(date_str, "%y%m%d")
            date_fmt = date_obj.strftime("%d-%b")
        except:
            date_fmt = ""

        rows.append({
            "date": date_fmt,
            "sample/junk": sample_type,
            "bottle": bottle,
            "file name alkalinity": file,  # add full file name
        })

# Build DataFrame
df_files = pd.DataFrame(rows)
df_files["First emf value"] = np.array(EMF_first)
# Sort by date then bottle

df_files['date_sort'] = pd.to_datetime(df_files['date'], format='%d-%b', errors='coerce')
df_files.sort_values(by=['date_sort', 'bottle'], inplace=True)
df_files.reset_index(drop=True, inplace=True)
df_files.drop(columns=['date_sort'], inplace=True)

# Update the existing Excel sheet
# Make sure we don’t overwrite other columns
# Assuming row count matches
for col in ["date", "sample/junk", "bottle", "file name alkalinity","First emf value"]:
    if col in df.columns:
        df[col] = df_files[col]
    else:
        df[col] = df_files[col]  # add new column if missing

# Save back
print(df)
df.to_excel(excel_file, index=False)

print(f"✅ Updated {len(df_files)} rows in {excel_file}")
