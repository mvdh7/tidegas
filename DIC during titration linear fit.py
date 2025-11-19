import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
from sklearn.linear_model import LinearRegression


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
plot_df = plot_df[plot_df["Mixing and waiting time (seconds)"]==4]


plot_df["Date"] = pd.to_datetime(plot_df["date"], format="%d/%m/%Y")

# # Filter from a certain date onwards
start_date = "10-10-2025"  # YYYY-MM-DD
plot_df = plot_df[plot_df["Date"] >= start_date]

#possible to select a specific date, or exclude a date


#plot_df = plot_df[plot_df["batch"]==1]

plot_df["acid added (mL)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["Calculated DIC (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"], errors='coerce')
plot_df["Percentage DIC (%)"] = pd.to_numeric(100*plot_df["Calculated DIC (umol/kg)"]/plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["DIC-loss (umol/kg)"] = pd.to_numeric(plot_df["Calculated DIC (umol/kg)"]-plot_df["Reference DIC (umol/kg)"], errors='coerce')
plot_df["waiting time (minutes)"] = pd.to_numeric(plot_df["waiting time (minutes)"], errors='coerce')
plot_df["Titration duration (seconds)"] = pd.to_numeric(plot_df["Titration duration (seconds)"], errors='coerce')

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
# plot_dic(plot_df, x="acid added (mL)", y="Calculated DIC (umol/kg)", use_colorbar=True)
# plot_dic(plot_df, x="acid added (mL)", y="Percentage DIC (%)", use_colorbar=True)
plot_dic(plot_df, x="acid added (mL)", y="DIC-loss (umol/kg)", use_colorbar=True)
# Example 2: With colorbar (continuous date hue)
# plot_dic(plot_df, x="Titration duration (seconds)", y="Calculated DIC (umol/kg)", use_colorbar=True)


plot_dic(plot_df, x="Titration duration (seconds)", y="Percentage DIC (%)", use_colorbar=True)
# Example 3: Another metric
plot_dic(plot_df, x="Titration duration (seconds)", y="DIC-loss (umol/kg)", use_colorbar=True)


def fit_dic_by_date(
    data,
    x_col,
    y_col,
    date_col="Date",
    min_points=5,
    figsize=(8,6),
    show_all_points=True
):
    """
    Linear fit of DIC metrics vs titration duration for each date with >= min_points.
    Plots fits and prints slope, intercept, and R² for each valid date.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe (must include x_col, y_col, and date_col).
    x_col : str
        Name of x variable (e.g., "Titration duration (seconds)").
    y_col : str
        Name of y variable (e.g., "DIC-loss (umol/kg)" or "Percentage DIC (%)").
    date_col : str
        Column with the date (converted to datetime before calling).
    min_points : int
        Minimum number of points required to compute a fit.
    figsize : tuple
        Size of output plot.
    show_all_points : bool
        If True, plot data points. If False, only show regression lines.
    """
    
    # Ensure data is a copy
    df = data.copy()
    
    # Convert date if necessary
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    
    # Drop missing values
    df = df.dropna(subset=[x_col, y_col, date_col])
    
    # Prepare figure
    plt.figure(figsize=figsize)

    if show_all_points:
        plt.scatter(df[x_col], df[y_col], alpha=0.2, label="All data", s=40)

    # Store results
    results = {}

    # Loop per unique date
    for date_value, group in df.groupby(date_col):
        if len(group) < min_points:
            continue  # Skip dates with too few points

        X = group[x_col].values.reshape(-1, 1)
        Y = group[y_col].values

        model = LinearRegression().fit(X, Y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, Y)

        # Save results
        results[date_value] = {"slope": slope, "intercept": intercept, "r2": r2}

        # Fit line for plotting
        x_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_fit = model.predict(x_fit)

        plt.plot(
            x_fit,
            y_fit,
            linewidth=2,
            label=f"{date_value.strftime('%d/%m/%Y')}  (R²={r2:.3f})"
        )

    # Labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Linear Fit: {y_col} vs {x_col} (per date)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return results


results_dic_loss = fit_dic_by_date(
    plot_df,
    x_col="Titration duration (seconds)",
    y_col="DIC-loss (umol/kg)"
)

results_dic_percentage = fit_dic_by_date(
    plot_df,
    x_col="Titration duration (seconds)",
    y_col="Percentage DIC (%)"
)
