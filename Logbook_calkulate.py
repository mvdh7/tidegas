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


#
# TODO add a check for increments, calk.read_dat does NOT work for single step titrations (or aborted runs)

# solution, build something myself or skip the ones where acid
# Quick fix skips them for now, but ideally new function
# it also does not work for low acid
acid_added = excel_df["acid added (mL)"]
acid_increment  = excel_df["acid increments (mL)"]
for i, row in dbs.iterrows():
    # datfile = os.path.join(row.file_path, row.file_name)
    datfile = row.file_path + '/' + row.file_name
    bakfile = datfile[:-3] + "bak"
    
    #single step titrations with one acid change create files that break calk.read_dat 
    if acid_added[i]/acid_increment[i]>1: 
    
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
# Initialize the variables used in calkulate from the excel file, the main log, takes the entire column
# TODO For future use potentially nice to split into two codes, one that takes everything and one that only calculates the last x dates to speed up computation
dbs["titrant_molinity"]  = excel_df["Titrant Molinity"]  # Extract directly from excel, default = 0.1
dbs["salinity"] =excel_df["Salinity"]  # Extract directly from excel, default = 30
dbs["temperature_override"] = excel_df["Temperature"]  # Uses the temperature from the logbook



#%%
#take the useful information from the DIC log and incorporate it into the main log 
# 1. Prepare DIC lookup table
dic_lookup = dic_df[["DIC file number","DIC file name","Sample type","Negative removed DIC (umol/L)","Daily reference DIC (umol/L)","File date"]].copy()

dic_lookup.rename(
    columns={
        "DIC file number": "Corresponding DIC file nr",
        "Negative removed DIC (umol/L)": "Calculated DIC (umol/L)",
        "Daily reference DIC (umol/L)": "Reference DIC (umol/L)",
    },
    inplace=True,
)

#use file nr column to get calculated DIC and store corresponding file name of DIC measurement with each titration 
excel_df["Calculated DIC (umol/L)"] = excel_df["Corresponding DIC file nr"].map(
    dic_lookup.set_index("Corresponding DIC file nr")["Calculated DIC (umol/L)"]
)
excel_df["file name DIC"] = excel_df["Corresponding DIC file nr"].map(
    dic_lookup.set_index("Corresponding DIC file nr")["DIC file name"]
)

# # Make sure dates are comparable
excel_df["Date"] = pd.to_datetime(excel_df["date"], format="%d/%m/%Y").dt.date
dic_lookup["Date"] = pd.to_datetime(dic_lookup["File date"], format="%d/%m/%Y").dt.date

# Now create lookup dictionaries keyed by (Date, Sample type)
ref_dic_map = dic_lookup.set_index(["Date", "Sample type"])["Reference DIC (umol/L)"].to_dict()


# Map them to excel_df using tuple keys (Date, sample/junk)
excel_keys = list(zip(excel_df["Date"], excel_df["sample/junk"]))

#add a default DIC of 2300 in case there is no Reference
excel_df["Reference DIC (umol/L)"] = [ref_dic_map.get(k, 2300) for k in excel_keys]

#TODO this also uses the DIC measured of the CRM as a reference, where we actually know the DIC
# perhaps it's better to have both the DIC from the Dickson analysis and the bobometer side by side
# now the alkalinity calculated is 'wrong' because the DIC is wrong
excel_df = excel_df.drop("Date",axis =1)


# -------------------------------------------------------------------
# 4. Convert to µmol/kg using seawater density
# -------------------------------------------------------------------
excel_df["Calculated DIC (umol/kg)"] = excel_df["Calculated DIC (umol/L)"] / calk.density.seawater_1atm_MP81(
    temperature=dbs.temperature_override,
    salinity=dbs.salinity,
)

# TODO is reference always taken at room temperature?!?!?!?!
#Convert the reference to umol/kg using density
excel_df["Reference DIC (umol/kg)"] = excel_df["Reference DIC (umol/L)"]/calk.density.seawater_1atm_MP81(
    temperature=21, salinity=dbs.salinity)


#%%
# prefereably used the calclated value, think of workflow, calculate reference DIC for each batch? 
dbs["dic"] = excel_df["Reference DIC (umol/kg)"]

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


