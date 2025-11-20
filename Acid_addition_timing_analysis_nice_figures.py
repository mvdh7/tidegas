import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.close('all')
#setup plotting parameters to make everything bigger
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 16,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 15,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })


#import csv file from local path
path = "data/vindta/r2co2/acid_timings_titration.csv"

#read into dataframe
df = pd.read_csv(path)


# Melt into long format
df_long = df.melt(id_vars=["acid added (mL)"], 
                  var_name="sample", 
                  value_name="timing")
# Extract day and bottle from column name
df_long[["day", "bottle"]] = df_long["sample"].str.split(" ", n=1, expand=True)
#ensure numerically increasing bottle numbers (and not 11 coming before 3 etc. )
df_long["bottle_num"] = df_long["bottle"].str.extract(r"(\d+)").astype(int)
# Reorder columns
df_long = df_long[["acid added (mL)", "day", "bottle_num", "timing"]]
df_long["acid added (mL)"] = df_long["acid added (mL)"]-0.15
df_long = df_long[0:1260]
#calculate mean and standard deviation
stats_raw = (
    df_long
    .groupby(["acid added (mL)", "day"])["timing"]
    .agg(["mean", "std", "count"])
    .reset_index())
measurement_noise = 2
def combine_std(row):
    if row["count"] == 1:
        return measurement_noise
    else:
        # combine real std with measurement uncertainty
        return np.sqrt(row["std"]**2 + measurement_noise**2)

stats_raw["std_total"] = stats_raw.apply(combine_std, axis=1)

stats = stats_raw.rename(columns={"std_total": "std_total"})[
    ["acid added (mL)", "day", "mean"]
]
stats["std_base"] = stats_raw["std"]
stats["std"] = stats_raw["std_total"]
#setup plotting
# Number of bottles you want colors for
n_bottles = df_long["bottle_num"].nunique()

# Generate colors from the plasma colormap
cmap =cm.get_cmap("plasma", n_bottles)  # 'plasma' is vibrant and perceptually uniform
colors = [cmap(i) for i in range(n_bottles)]
# Map bottle numbers to colors
bottle_colors = {bottle_num: colors[i] for i, bottle_num in enumerate(sorted(df_long["bottle_num"].unique()))}

def plot_day(df_long, day):
    """Plot all bottles for a given day with maximally distinct colors + markers."""
    
    subset = df_long[df_long["day"] == str(day)]
    bottles = sorted(subset["bottle_num"].unique())
    n_bottles = len(bottles)

    # --- Colors: maximally distinct from plasma colormap ---
    cmap = cm.get_cmap("plasma")
    positions = np.linspace(0.05, 0.95, n_bottles)
    bottle_colors = {b: cmap(pos) for b, pos in zip(bottles, positions)}

    # --- Markers: distinct shapes ---
    markers = ["o", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H","p"]
    bottle_markers = {b: markers[i % len(markers)] for i, b in enumerate(bottles)}

    plt.figure(figsize=(9,6))

    # Plot each bottle
    for bottle_num, group in subset.groupby("bottle_num"):
        color = bottle_colors[bottle_num]
        marker = bottle_markers[bottle_num]

        plt.scatter(
            group["acid added (mL)"], group["timing"],
            label=f"Bottle {bottle_num}", color=color,
            marker=marker, s=80, alpha=1)

        plt.errorbar(
            group["acid added (mL)"], group["timing"],
            yerr=np.ones(len(group))*2,
            fmt="none", ecolor=color, capsize=4, alpha=0.7
        )

    # Mean ± std for the day
    stats_day = subset.groupby("acid added (mL)")["timing"].agg(["mean", "std"])
    plt.plot(
        stats_day.index, stats_day["mean"],
        color="black", linestyle="--", linewidth=2,
        marker="s", markersize=6, label="Mean ± std"
    )
    plt.fill_between(
        stats_day.index,
        stats_day["mean"] - stats_day["std"],
        stats_day["mean"] + stats_day["std"],
        color="gray", alpha=0.7
    )

    plt.title(f"Titration step duration vs Acid Added on {day[:2]}-{day[2:]} for various bottles", fontsize=18)
    plt.xlabel("Acid added (mL)", fontsize=16)
    plt.ylabel("Time (s)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_days(stats, df_long, days):
    """Compare mean ± std curves between selected days, using maximally distinct colors."""

    plt.figure(figsize=(9,6))

    # Number of days being plotted
    n_days = len(days)

    # Generate maximally distinct colors from plasma
    cmap = cm.get_cmap("plasma")
    positions = np.linspace(0.05, 0.95, n_days)
    day_colors = {day: cmap(p) for day, p in zip(days, positions)}
    # Highly distinguishable markers, cycling if needed
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H"]
    day_markers = {day: markers[i % len(markers)] for i, day in enumerate(days)}
  
    for day in days:
        subset = stats[stats["day"] == str(day)]
    
        # ---- FIX DUPLICATE X-VALUES ----
        subset = (
            subset.groupby("acid added (mL)")[["mean", "std"]]
            .mean()
            .reset_index()
            .sort_values("acid added (mL)")
        )
        n_bottles = df_long[df_long["day"] == str(day)]["bottle_num"].nunique()
        # if n_bottles ==1:
        #     continue
        label = f"{day[:2]}-{day[2:]} (N={n_bottles})"
        color = day_colors[day]
        marker = day_markers[day]
        
        # Mean curve
        plt.plot(subset["acid added (mL)"], subset["mean"],
                 marker=marker, linestyle="-", linewidth=2, label=label, color=color)

        # Shaded std region
        plt.fill_between(subset["acid added (mL)"],
                         subset["mean"] - subset["std"],
                         subset["mean"] + subset["std"],
                         color=color, alpha=0.25)

    plt.xlabel("Acid added (mL)", fontsize=16)
    plt.ylabel("Time (mean ± std) (s)", fontsize=16)
    plt.title("Comparison of Titration Step Duration Across Days", fontsize=18)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

#%%
days_measured = ["1609","1709","1809","3009","0110","0310","0610","1710","2910","3010", "0511",  "1011","1211"]
plot_all_individual = True

# if plot_all_individual:
#     for i,date in enumerate(days_measured):
#         print(date)        
#         plot_day(df_long, date)

#plot a specific day
plot_day(df_long,"1709")

compare_days(stats,df_long, days_measured)
#compare specific days
#compare_days(stats, df_long,[1609, 1709,1809])
