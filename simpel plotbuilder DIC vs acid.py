import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

#setup plotting parameters to make everything bigger
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 18,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 18,    # x tick labels
    "ytick.labelsize": 18,    # y tick labels
    "legend.fontsize": 16,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure columns exist
if not all(col in df.columns for col in ["Calculated DIC (umol/kg)", "acid added (mL)", "date"]):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = df.dropna(subset=["Calculated DIC (umol/kg)", "acid added (mL)", "date", "waiting time (minutes)"]).copy()

#select only the files without any (significant) waiting time 
plot_df =plot_df[plot_df["waiting time (minutes)"]<=0.05]
plot_df =plot_df[plot_df["acid increments (mL)"]<=0.15]
plot_df = plot_df[plot_df["date"]!="09/10/2025"]



plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "10-11-2025"  # YYYY-MM-DD
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date


#plot_df = plot_df[plot_df["batch"]==1]

plot_df["acid added (mL)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Percentage DIC"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')
#%%
# Scatter plot with hue by date
plt.figure()
sns.scatterplot(
    data=plot_df,
    x="acid added (mL)",
    y="Percentage DIC",
    hue="date",
    palette="tab20",
    s=70
)
plt.xlabel("Acid added (mL)")
plt.ylabel("Remaining DIC (%)")
plt.title("DIC vs Acid Added for Different Days")
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%%
# Scatter plot with hue by date
plt.figure()
sns.scatterplot(
    data=plot_df,
    x="acid added (mL)",
    y="Calculated DIC (umol/kg)",
    hue="date",
    palette="tab20",
    s=70
)

plt.xlabel("Acid added (mL)")
plt.ylabel("Remaining DIC (umol/kg)")
plt.title("DIC vs Acid Added for Different Days")
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%%
# Scatter plot with hue by date
plt.figure()
sns.scatterplot(
    data=plot_df,
    x="Titration duration (seconds)",
    y="Calculated DIC (umol/kg)",
    hue="date",
    palette="tab20",
    s=70
)

plt.xlabel("Titration duration (seconds)")
plt.ylabel("Remaining DIC ")
plt.title("DIC vs Acid Added for Different Days")
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
# Scatter plot with hue by date
plt.figure()
sns.scatterplot(
    data=plot_df,
    x="Titration duration (seconds)",
    y="Percentage DIC",
    hue="date",
    palette="tab20",
    s=70
)

plt.xlabel("Titration duration (seconds)")
plt.ylabel("Remaining DIC (%) ")
plt.title("DIC vs Acid Added for Different Days")
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
plt.figure()

# Create the scatter manually using matplotlib for colorbar control
scatter = plt.scatter(
    plot_df["Titration duration (seconds)"],
    plot_df["Calculated DIC (umol/kg)"],
    c=plot_df["Date"].map(mdates.date2num),  # map dates to numbers
    cmap="viridis",  # or "plasma", "cividis", etc.
    s=70
)

plt.xlabel("Titration duration (seconds)")
plt.ylabel("Calculated DIC (µmol/kg)")
plt.title("DIC vs Acid Added Over Time")

# Add a colorbar with formatted date ticks
cbar = plt.colorbar(scatter)
cbar.ax.set_ylabel("Date")
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))

plt.tight_layout()
plt.show()