import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os 
#bobometer analysis
#This script loads an individual DIC measurement file 
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 16,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 15,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })

folder_path = "data/vindta/r2co2/Bobometer"
file_path = "data/vindta/r2co2/Bobometer/co2data (15).txt"



file = pd.read_csv(file_path)
plt.figure()

plt.plot(file[" Cell[uAs]"][200:])
plt.ylabel("integrated current [uAs]")
plt.xlabel("time in [s]")

#%%
plt.figure()

plt.plot(file[" Cell[uAs]"][100:200])
plt.ylabel("integrated current [uAs]")
plt.xlabel("time in [s]")
#%%

tfiles = os.listdir(folder_path)

tfiles.remove("config (1).ini")


counter = 0 

files = []
int_curr = []
plt.figure()
for file in tfiles:
    files.append(file)
    integrated_current = pd.read_csv(folder_path + "/"+file)[" Cell[uAs]"]
    int_curr.append(np.array(integrated_current)[-1])
    print(file)
    integrated_current_normalized = integrated_current/np.average(integrated_current[-180:])
    if max(integrated_current_normalized) <= 199990.2:
        plt.plot(integrated_current_normalized[-180:], label = f"{file[-7:-4]}")
        counter +=1
        
#plt.ylim(0.998,1.002)
plt.ylabel("Normalized integrated current [final 3 minutes]")
plt.xlabel("time in [s]")
plt.tight_layout()


