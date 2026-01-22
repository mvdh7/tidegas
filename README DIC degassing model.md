# Modelling readme

## Introduction
Two approaches to modelling DIC were explored. 
In both approaches DIC data corrected to in-situ DIC was fitted to create a DIC model. There is a global approach and a daily approach. 

## Daily approach
The daily approach uses the DIC measurements on a specific day to create a DIC degassing model. In these models the variations of titrant duration across days are reduced. (Titration duration does still vary throughout the day though, as can be seen from plotting the titration duration against the titrant volume). Two examples are provided in the scripts:
"Plotting impact of variable DIC 30_10 measurement"
"Plotting impact of variable DIC 11_11 measurement"

In these scripts the fit is calculated, and the alkalinity is recalculated with the fit (the degassing model) and with a shifted degassing model, which shifts the degassing model such that it starts at the theoretical DIC prediction form alkalinity and pH. 

## Global approach
The global approach combines data from multiple days to make a general DIC degassing model. Some dates have been excluded, as the coulometer was not functioning properly during this period. The general period is selected such that titration durations are similar. 
The global model for DIC is calculated in the script:
Global DIC model builder, which creates an global_dic_degassing_model and applied in:
Global DIC alkalinity recalculated. 

The impact of adding or removing certain samples can be studied in 
Plotting all titrations in situ DIC vs acid with correction and fit