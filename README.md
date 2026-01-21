# Titration degassing

## Introduction
Tidegas is a project to determine the degassing of CO2 and the impact of degassing on alkalinity during open cell alkalinity titrations performed on the VINDTA 13C. The Calkulate software developed my Matthew Humphreys (https://github.com/mvdh7/calkulate) allows for a variable DIC (dissolved inorganic carbon, which dissolved CO2 is part of) input which does include degassing. 

## Degassing of CO2
The degassing of CO2 and the resulting reduction of DIC was measured using a VINDTA 13C and a coulometric cell. Titrations on the VINDTA with varying amount of titrant volumes were performed. The remaining DIC at the end of the titration was measured with the coulometer. 

## Raw data
Data from the experiments is stored in data/vindta/r2co2
The Bobometer folder contains all the measurement data from the coulometer, renamed to display the date and index
The Bobometer raw data contains all raw data. 
The Nico folder contains the data from all individual titration measurements
The Nico.dbs contains the database from the VINDTA measurements and

## Processed DIC data
The processed DATA for the DIC is stored in DIC_logbook, which is created by the DIC_logbook_creator. This creator corrects DIC meaurements for negative currents. In the logbook relevant measurement data is stored and calculated. A daily reference is calculated for each sample type daily, including measurement with a 1 in the reference column. 

## Processed VINDTA data
The processed data from the VINDTA is stored as the VINDTA_logbook. 
The metadata from the dbs file can be extracted using logbook.extract_data, which also creates backups
The initial processing of the data was done by logbook_calkulate. This writes to Logbook_automated_by_python_testing, which is also used for plotting. The logbook contains several values for DIC, which should not be confused:
Calculated DIC (umol/L) is the DIC as obtained from the coulometer, with negative currents removed 
Calculated DIC (umol/kg) is the DIC obtained from the coulometer /kg, using the temperature from the logbook and salinity to calculate the density
Reference DIC (umol/L) is the daily reference as calculated in the DIC_logbook, which is unique for each sample type. 
Reference DIC (umol/kg) is the daily reference converted to /kg, using a FIXED! temperature (21C) and salinity to calculate

Calculated in situ DIC (umol/kg) is the calculated DIC corrected for the measuring delay and initialisation and sampling phase of the coulometer. The measuring delay is stored in the logbook as waiting time (minutes), while the initialisation and sampling phase was determined to last 95 seconds. An expenonential decay model is used to calculate the in situ DIC values in the script "DIC back calculation for all titrations".

The decay model parameters are obtained from experiments (the degassing rate k) and theoretical prediction from the carbonate system. For the theoretical predictions daily reference alkalinity measurements are used, which are indicated with a 1 in column "Alkalinity daily reference measurement". For in-situ DIC values the (DIC equilibrium) C value from the corresponding daily reference titration are stored in the "C_from_reference (%)" column. 


## Plotting and analysis
Several plotting scripts are made available, which use the Logbook_automated_by_python_testing data
