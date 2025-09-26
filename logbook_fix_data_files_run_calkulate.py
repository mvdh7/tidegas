# %%
import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)
excel_df = pd.read_excel("Logbook_automated_by_python_testing_2.xlsx")

tfiles = os.listdir(file_path)



acid_increment  = excel_df["acid increments (mL)"]
for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    bakfile = datfile[:-3] + "bak"
    dat_data = calk.read_dat(datfile)
    
    
    calk.write_dat(
        datfile,
        # The line below sets the correct titrant_amount values
        np.arange(0, len(dat_data.titrant_amount) * acid_increment[i], acid_increment[i]),
        # -----------------------------------------------------
        dat_data.measurement,
        dat_data.temperature,
        mode="w",
        line0=f"Based on '{bakfile} with acid increment {acid_increment[i]} mL'",
    )


#%%
# Initialize the variables used in calkulate from the excel file 
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30
dbs["dic"] = ( excel_df["Calculated DIC ug/kg"]
    .combine_first(excel_df["raw DIC ug/L"])
    .combine_first(excel_df["Placeholder DIC"]))    #take fpreferably the calculated DIC, but if not, take the next best or the placeholder (set to 2300 ish)
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

#%%

# TESTING CODE
# goal is to automatically update calkulated alkalinity in logbook file

# method is to take elements from the DBS, and assign them to specific columns based on file name in the logbook

# bonus is adding the first measurement of EMF as well

reduced_dbs = dbs[["file_name","alkalinity","emf0","pH_init"]]
test_alkalinity = np.array(dbs["alkalinity"])
print(test_alkalinity)

# Prepare reduced dataframe with matching column names
update_df = reduced_dbs[["alkalinity", "emf0"]]
update_df.rename(columns={"alkalinity": "calkulate alkalinity",
    "emf0": "emf0 value"
}, inplace=True)

# Update in place
excel_df.update(update_df)

# Save back
excel_df.to_excel("Logbook_automated_by_python_testing_2.xlsx", index=False)


