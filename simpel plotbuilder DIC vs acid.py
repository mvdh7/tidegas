import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#setup plotting parameters to make everything bigger
plt.rcParams.update({         # Set standard fontsizes for plot labels
    "axes.labelsize": 16,     # x and y axis labels
    "axes.titlesize": 18,     # plot title
    "xtick.labelsize": 16,    # x tick labels
    "ytick.labelsize": 16,    # y tick labels
    "legend.fontsize": 15,    # legend text
    "figure.titlesize": 20,   # figure title if used
     })
# Read your Excel file
excel_file = "logbook_automated_by_python_testing.xlsx"
df = pd.read_excel(excel_file)

# Make sure columns exist
if not all(col in df.columns for col in ["raw DIC umol/L", "acid in mL extracted", "date"]):
    raise ValueError("Required columns missing in the Excel file.")

# Drop rows with missing data in the columns we need
plot_df = df.dropna(subset=["raw DIC umol/L", "acid added (mL)", "date"]).copy()
plot_df["acid added (mL)"] = pd.to_numeric(plot_df["acid added (mL)"], errors='coerce')
plot_df["raw DIC umol/L"] = pd.to_numeric(plot_df["raw DIC umol/L"], errors='coerce')
# Set Seaborn style
sns.set(style="whitegrid")

# Scatter plot with hue by date
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=plot_df,
    x="acid added (mL)",
    y="raw DIC umol/L",
    hue="date",
    palette="tab10",
    s=70
)

plt.xlabel("Acid added (mL)")
plt.ylabel("Raw DIC (µmol/L)")
plt.title("Raw DIC vs Acid Added for Different Days")
plt.legend(title="Date", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
