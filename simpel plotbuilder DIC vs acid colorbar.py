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
excel_file = "logbook_in_situ_corrected_testing.xlsx"
excel_file = "C:/Users/nicor/OneDrive/Documenten/GitHub/tidegas/logbook_in_situ_corrected.xlsx" 
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
plot_df = plot_df[plot_df["Mixing and waiting time (seconds)"]==4]


plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "10-10-2025"  # YYYY-MM-DD
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date


#plot_df = plot_df[plot_df["batch"]==1]

plot_df["Titrant Volume (ml)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Calculated in situ DIC (umol/kg)"] =  pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"], errors='coerce')
plot_df["DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC (%)"] = pd.to_numeric(100*plot_df["Calculated in situ DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["In-situ DIC difference (umol)"] = pd.to_numeric(plot_df["Calculated in situ DIC (umol/kg)"]-plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting Time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration Duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')

def plot_dic(
    data,
    x,
    y,
    use_colorbar=False,
    cmap="viridis",
    palette="tab20",
    s=70,
    dayfirst=True
):
    """
    Create a scatter plot of DIC data with either a discrete hue legend or a continuous colorbar for date.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the columns x, y, and 'date' (or 'Date').
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    use_colorbar : bool, optional
        If True, use continuous colorbar for date instead of categorical hue.
    cmap : str, optional
        Colormap for colorbar plotting.
    palette : str, optional
        Palette for seaborn categorical hue plotting.
    s : int, optional
        Marker size.
    dayfirst : bool, optional
        Whether to interpret day-first date format.
    """

    # Normalize column name capitalization
    date_col = "Date"

    # Convert to datetime
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col], dayfirst=dayfirst)

    plt.figure(figsize=(7, 5))

    if use_colorbar:
        # Continuous color scale using Matplotlib
        scatter = plt.scatter(
            data[x],
            data[y],
            c=data[date_col].map(mdates.date2num),
            cmap=cmap,
            s=s
        )

        cbar = plt.colorbar(scatter)
        cbar.ax.set_ylabel("Date")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))

    else:
        # Discrete color scale using Seaborn
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=date_col,
            palette=palette,
            s=s
        )
        plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Labels and formatting
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} vs {x} by Date")
    plt.tight_layout()
    plt.show()
    
    
# # Example 1: With legend (categorical hue)
plot_dic(plot_df, x="Titrant Volume (ml)", y="Calculated DIC (umol/kg)", use_colorbar=True)

plot_dic(plot_df, x="Titrant Volume (ml)", y="Calculated in situ DIC (umol/kg)", use_colorbar=True)

plot_dic(plot_df, x="Titrant Volume (ml)", y="In-situ DIC (%)", use_colorbar=True)

plot_dic(plot_df, x="Titrant Volume (ml)", y="In-situ DIC difference (umol)", use_colorbar=True)


plot_dic(plot_df, x="Titration duration (seconds)", y="In-situ DIC difference (umol)", use_colorbar=True)
#%%
plot_dic(plot_df, x="Titration duration (seconds)", y="Calculated DIC (umol/kg)", use_colorbar=True)
plot_dic(plot_df, x="Titration duration (seconds)", y="Calculated in situ DIC (umol/kg)", use_colorbar=True)
plot_dic(plot_df, x="Titration duration (seconds)", y="In-situ DIC (%)", use_colorbar=True)

#%%
# plot_dic(plot_df, x="acid added (mL)", y="Percentage DIC (%)", use_colorbar=True)
plot_dic(plot_df, x="acid added (mL)", y="DIC-loss (umol/kg)", use_colorbar=True)
# Example 2: With colorbar (continuous date hue)
# plot_dic(plot_df, x="Titration duration (seconds)", y="Calculated DIC (umol/kg)", use_colorbar=True)


plot_dic(plot_df, x="Titration duration (seconds)", y="Percentage DIC (%)", use_colorbar=True)
# Example 3: Another metric
plot_dic(plot_df, x="Titration duration (seconds)", y="DIC-loss (umol/kg)", use_colorbar=True)