# %%
import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

excel_file = "logbook_automated_by_python_testing.xlsx"

excel_df = pd.read_excel(excel_file)

tfiles = os.listdir(file_path)


#this script loads the logbook and transforms it into a dataframe
#From this dataframe input parameters are extracted, and output is saved in the logbook
#inputs: acid increments (mL), Titrant Molinity, Salinity, Calculated DIC ug/kg, raw DIC ug/L, Placeholder DIC


# the column acid increments (mL) is used to update the datfile, using the correct spacing in between datapoints
# TODO Ensure that the last value is also added to the datfile, making a new backup?


# TODO add a check for increments, calk.read_dat does NOT work for single step titrations (or aborted runs)
# solution, build something myself or skip the ones where acid
acid_increment  = excel_df["acid increments (mL)"]
for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    bakfile = datfile[:-3] + "bak"
    if acid_increment[i] <=1:
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
print('passed')
# Initialize the variables used in calkulate from the excel file, takes the entire column
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30

# ???DIC values are different, some measurements might not have corresponding DIC values
# ??? for the alkalinity measurement the DIC at 0 (if available, else from the same batch?) acid added should be taken?
# prefereably used the calclated value, think of workflow, calculate reference DIC for each batch? 
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]
dbs["temperature_override"] = excel_df["Temperature"]  # Uses the temperature from the logbook

# Set bad files to ignore
dbs["file_good"] = ~dbs.bottle.isin(["junk-250916-01","junk-250916-08","junk-250930-01"])
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
excel_df.to_excel(excel_file, index=False)


