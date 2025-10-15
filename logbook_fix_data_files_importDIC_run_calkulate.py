# %%
import os
import shutil
import pandas as pd
import calkulate as calk
import numpy as np

# Import dbs file
file_path = "data/vindta/r2co2/Nico"
dbs = calk.read_dbs("data/vindta/r2co2/Nico.dbs", file_path=file_path)

DIC_file = "DIC_logbook.csv"
excel_file = "logbook_automated_by_python_testing.xlsx"

dic_df = pd.read_csv(DIC_file)
excel_df = pd.read_excel(excel_file)

tfiles = os.listdir(file_path)


#this script loads the logbook and transforms it into a dataframe
#From this dataframe input parameters are extracted, and output is saved in the logbook
#inputs: acid increments (mL), Titrant Molinity, Salinity, Calculated DIC ug/kg, raw DIC ug/L


# the column acid increments (mL) is used to update the datfile, using the correct spacing in between datapoints


# TODO add a check for increments, calk.read_dat does NOT work for single step titrations (or aborted runs)
# solution, build something myself or skip the ones where acid
# Quick fix skips them for now, but ideally new function
acid_increment  = excel_df["acid increments (mL)"]
for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    bakfile = datfile[:-3] + "bak"
    if acid_increment[i] <=1: #large acid increments create files that break calk.read_dat 
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
# Initialize the variables used in calkulate from the excel file, takes the entire column
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30
dbs["temperature_override"] = excel_df["Temperature"]  # Uses the temperature from the logbook



#TODO Ideally want to take the DIC reference from the DIC excel file, especially if it is varying through time 



#Convert the reference to umol/kg using density
excel_df["Reference DIC (umol/kg)"] = excel_df["Reference DIC (umol/L)"]/calk.density.seawater_1atm_MP81(
    temperature=dbs.temperature_override, salinity=dbs.salinity)


#%%
# prefereably used the calclated value, think of workflow, calculate reference DIC for each batch? 
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]


#%%
dic_lookup = dic_df[["DIC file number", "Negative removed DIC (umol/L)"]].copy()
dic_lookup.rename(
    columns={
        "DIC file number": "Corresponding DIC file nr",
        "Negative removed DIC (umol/L)": "Calculated DIC (umol/L)",
    },
    inplace=True,
)

# # Ensure numeric file numbers (important for clean merging)
# dic_lookup["Corresponding DIC file nr"] = pd.to_numeric(dic_lookup["Corresponding DIC file nr"], errors="coerce")
# excel_df["Corresponding DIC file nr"] = pd.to_numeric(excel_df["Corresponding DIC file nr"], errors="coerce")

# -------------------------------------------------------------------
# 3. Merge the calculated DIC values into your excel_df
# -------------------------------------------------------------------
excel_df["Calculated DIC (umol/L)"] = excel_df["Corresponding DIC file nr"].map(
    dic_lookup.set_index("Corresponding DIC file nr")["Calculated DIC (umol/L)"]
)
# -------------------------------------------------------------------
# 4. Convert to µmol/kg using seawater density
# -------------------------------------------------------------------
excel_df["Calculated DIC (umol/kg)"] = excel_df["Calculated DIC (umol/L)"] / calk.density.seawater_1atm_MP81(
    temperature=dbs.temperature_override,
    salinity=dbs.salinity,
)



#%%
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


