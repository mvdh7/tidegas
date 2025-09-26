# %%
import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

# For each titration file, if there isn't already a backup copy:
# 1. Make a backup copy (with extension .bak).
# TODO 2. Extract the value of the first EMF reading and store it (for adding it in the log later)
# 3. Open the .dat, replace first column with correct volumes, TODO directly extracted form the log, and save.
# (because R2-CO2 just saves zeroes in the titrant_amount column!)


tfiles = os.listdir(file_path)

#store a list of ALL first EMF values, which can be added to the dataframe later
EMF_first = []

for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    bakfile = datfile[:-3] + "bak"
    dat_data = calk.read_dat(bakfile)
    
    EMF_first.append(dat_data.measurement[0])
    if row.file_name[:-3] + "bak" not in tfiles:
        print(row.file_name)
        shutil.copyfile(datfile, bakfile)
        
        calk.write_dat(
            datfile,
            # The line below sets the correct titrant_amount values
            np.arange(0, len(dat_data.titrant_amount) * 0.15, 0.15),
            # -----------------------------------------------------
            dat_data.measurement,
            dat_data.temperature,
            mode="w",
            line0=f"Based on '{bakfile}'",
        )
        

# Now we can proceed with Calkulate as normal
dbs["titrant_molinity"] = 0.1  # guess for now, calibrate later
dbs["salinity"] = 30  # guess for now, measure later
dbs["dic"] = 2350  # guess for now, measure later
dbs["temperature_override"] = 25  # as the files record 0 °C

# Set bad files to ignore
dbs["file_good"] = ~dbs.bottle.isin(["junk-250916-01","junk-250916-08"])
dbs.solve()

# TODO Workaround for density storing bug in v23.7.0
dbs["analyte_mass"] = (
    1e-3
    * dbs.analyte_volume
    / calk.density.seawater_1atm_MP81(
        temperature=dbs.temperature_override, salinity=dbs.salinity
    )
)

# Extract one titration to plot
tt = dbs.to_Titration(14)
tt.plot_pH()
tt.plot_components()
tt.plot_alkalinity()

#%%

# TESTING CODE
# goal is to automatically update calkulated alkalinity in logbook file

# method is to take elements from the DBS, and assign them to specific columns based on file name in the logbook

# bonus is adding the first measurement of EMF as well

reduced_dbs = dbs[["file_name","alkalinity","emf0","pH_init"]]
reduced_dbs["EMF first"] = np.array(EMF_first)
test_alkalinity = np.array(dbs["alkalinity"])
print(test_alkalinity)
print(tt.file_name[:-4])

excel_df = pd.read_excel("Logbook_automated_by_python_testing.xlsx").set_index("Alkalinity file name")

# Prepare reduced dataframe with matching column names
update_df = reduced_dbs.set_index("file_name")[["alkalinity", "EMF first", "emf0"]]
update_df.rename(columns={
    "alkalinity": "calkulate alkalinity",
    "EMF first": "First emf value",
    "emf0": "emf0 value"
}, inplace=True)

# Update in place
excel_df.update(update_df)

# Save back
excel_df.to_excel("Logbook_automated_by_python_testing.xlsx", index=False)


