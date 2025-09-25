import pandas as pd
import matplotlib.pyplot as plt

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
path = "data/vindta/r2co2/Nico/acid_timings_titration.csv"

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


#calculate mean and standard deviation
stats = (
    df_long
    .groupby(["acid added (mL)", "day"])["timing"]
    .agg(["mean", "std"])
    .reset_index()
)

#setup plotting

def plot_day(df_long, day):
    """Plot all bottles for a given day with mean ± std."""
    subset = df_long[df_long["day"] == str(day)]
    x = subset["acid added (mL)"].unique()
    
    plt.figure(figsize=(8,6))
    
    # Sort bottles numerically
    for bottle_num, group in subset.groupby("bottle_num"):
        label = f"bottle {bottle_num}"
        plt.plot(group["acid added (mL)"], group["timing"], 
                 marker="o", label=label,alpha =0.5)
    
    # Mean ± std for this day
    stats = subset.groupby("acid added (mL)")["timing"].agg(["mean","std"])
    plt.plot(stats.index, stats["mean"], color="black", marker="s", 
             linestyle="--", linewidth=2, label="Mean")
    plt.errorbar(stats.index, stats["mean"], yerr=stats["std"], 
                 fmt="none", ecolor="gray", capsize=5, alpha=0.7)
    
    # Add sample size N
    n_bottles = subset["bottle_num"].nunique()
    plt.title(f"Day {day} - Timings vs Acid Added (N={n_bottles})")
    
    plt.xlabel("Acid added (mL)")
    plt.ylabel("Timing")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def compare_days(stats, df_long, days):
    """Compare mean ± std curves between selected days."""
    plt.figure(figsize=(8,6))
    
    for day in days:
        subset = stats[stats["day"] == str(day)]
        
        # Sample size for this day
        n_bottles = df_long[df_long["day"] == str(day)]["bottle_num"].nunique()
        label = f"Day {day} (N={n_bottles})"
        
        plt.plot(subset["acid added (mL)"], subset["mean"], 
                 marker="s", label=label)
        plt.fill_between(subset["acid added (mL)"],
                         subset["mean"] - subset["std"],
                         subset["mean"] + subset["std"],
                         alpha=0.2)
    
    plt.xlabel("Acid added (mL)")
    plt.ylabel("Timing (mean ± std)")
    plt.title("Comparison between days")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

#%%
days_measured = [1609,1709,1809]
plot_all_individual = True

if plot_all_individual:
    for i,date in enumerate(days_measured):
        print(date)        
        plot_day(df_long, date)

#plot a specific day
#plot_day(df_long,1709)

compare_days(stats,df_long, days_measured)
#compare specific days
#compare_days(stats, df_long,[1609, 1709,1809])
