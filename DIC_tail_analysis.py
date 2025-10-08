import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os 
#bobometer analysis
#This script loads an individual DIC measurement file 
plt.close('all')
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 16,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 15,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })

folder_path = "data/vindta/r2co2/Bobometer"


file_path = "data/vindta/r2co2/Bobometer/co2data (63).txt"
file = pd.read_csv(file_path)






#%%
#first we have to correct for insanely high peaks that skew everything
#we just take the average of the next two points
#putting the mask at 1e7 should allow for all DIC data to pass except peaks
file[" Cell[uAs]"] = file[" Cell[uAs]"].mask(file[" Cell[uAs]"] >= 1e7,
    (file[" Cell[uAs]"].shift(1) + file[" Cell[uAs]"].shift(-1)) / 2)


total_set_point = file[" set_Cell[mA]"].cumsum() * 1000
total_cell_integrated = file[" Cell[uAs]"]
Total_cell = file[" Cell[mA]"].cumsum() * 1000



plt.figure()
plt.title(f"Example file nr. {file_path[-8:-4]}")
plt.plot(total_cell_integrated, label = "Total cell from integration")
plt.ylabel("Integrated current [uAs]")
plt.xlabel("time in [s]")
plt.tight_layout()
#Now we can loop over the values and simply refuse to let the integrated current decrease, only adding positive differences 
total_cell_negative_removed = [0]

integrated_current = 0
counter = 0


for i in range(len(total_cell_integrated)-1):
    
    if file[" Cell[uAs]"][i+1]<= file[" Cell[uAs]"][i]:
        counter += 1
    else: 
        integrated_current += file[" Cell[uAs]"][i+1]- file[" Cell[uAs]"][i]
        
    total_cell_negative_removed.append(integrated_current)

#taking the difference shows how much and where there is a negative drift
negative_removed_difference = total_cell_negative_removed-file[" Cell[uAs]"]



#%%
plt.figure()
plt.title("Integrated Cell Current – Raw vs. Cleaned Data")
plt.plot(file[" Cell[uAs]"], label="Raw data (spike-filtered)")
plt.plot(total_cell_negative_removed, label="Negative current removed")
plt.ylabel("Integrated current [µA·s]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.legend()

#%%
plt.figure()
plt.title("Change in Integrated Current After Negative Removal")
plt.plot(negative_removed_difference, label="Δ Current after removing negative slopes")
plt.ylabel("Change in integrated current [µA·s]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.legend()

#%%
# Create difference-only array and corresponding linspace
negative_removed_difference_only = negative_removed_difference[negative_removed_difference.diff() != 0]
negative_difference_linspace = np.linspace(0, len(negative_removed_difference_only)+1, len(negative_removed_difference_only))

plt.figure()
plt.title("Filtered Current Changes (Only Non-Zero Differences)")
plt.plot(negative_difference_linspace, negative_removed_difference_only, label="Non-zero Δ current")
plt.ylabel("Change in integrated current [µA·s]")
plt.xlabel("Index (filtered time steps)")
plt.tight_layout()
plt.legend()

#%%
# Compute slope
slope = negative_removed_difference[negative_removed_difference.diff() > 0.0001].diff()

plt.figure()
plt.title("Slope of Integrated Current (Positive Changes Only)")
plt.plot(-slope, label="Slope (negative direction)")
plt.ylabel("Rate of change [µA·s per step]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.legend()

plt.figure()
plt.title("Slope of Integrated Current (Scatter Representation)")
plt.scatter(slope.index, -slope, label="Slope (negative direction)")
plt.ylabel("Rate of change [µA·s per step]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.legend()

total_negative = np.sum(slope)
